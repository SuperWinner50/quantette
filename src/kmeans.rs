//! Color quantization by k-means clustering.
//!
//! The current single-threaded implementation uses online k-means (also known as MacQueen's k-means).
//! The current multi-threaded implementation uses minibatch k-means.
//! The online nature of these implementations allows them to escape from local minima more easily
//! compared to batch k-means (a.k.a. Lloyd's algorithm).
//!
//! Both implementations incorporate a learning rate (per centroid) that
//! decreases the influence of each successive sample to the same centroid.
//! So, increasing the number of samples has diminishing returns.
//! Rather, these methods should only need to make one pass (or even less) over the input data.
//!
//! # Parameters
//! Here are explanations for the parameters taken by the quantization functions in this module:
//! - `num_samples`: the number of pixels/colors to sample.
//!     More samples will give more accurate results but with diminishing returns.
//!     Typically, this is less than or equal to the number of colors in the input.
//! - `initial_centroids`: the initial colors to refine into a color palette.
//!     These centroids should be a decent representation of the input colors to prevent
//!     weird results and/or k-means getting stuck in a bad local minima.
//!     You can use palettes generated by [Wu's color quantizer](crate::wu) as the initial centroids.
//! - `seed`: the seed value for the random number generator (for reproducible results).
//! - `batch_size`: used only for the parallel quantization functions (needs the `threads` feature).
//!     This sets the number of samples to batch together when updating the centroids.
//!     Smaller batch sizes give higher quality results while larger batch sizes
//!     take less time to run (but with diminishing returns).
//!     Each batch will be distributed evenly across the threads in the current [`rayon`] thread pool.

// The k-means implementations here are based upon the following paper:
//
// Thompson, S., Celebi, M.E. & Buck, K.H. Fast color quantization using MacQueen’s k-means algorithm.
// Journal of Real-Time Image Processing, vol. 17, 1609–1624, 2020.
// https://doi.org/10.1007/s11554-019-00914-6
//
// Accessed from https://faculty.uca.edu/ecelebi/documents/JRTIP_2020a.pdf

use crate::{
    AboveMaxLen, ColorComponents, ColorCounts, ColorCountsRemap, PaletteSize, QuantizeOutput,
    MAX_COLORS,
};

#[cfg(feature = "threads")]
use crate::ColorCountsParallelRemap;

use std::{array, marker::PhantomData, ops::Deref};

use num_traits::AsPrimitive;
use palette::cast::{self, AsArrays};
use rand::{prelude::Distribution, SeedableRng};
use rand_distr::{weighted_alias::WeightedAliasIndex, Uniform};
use rand_xoshiro::Xoroshiro128PlusPlus;
use wide::{f32x8, u32x8, CmpLe};

#[cfg(feature = "threads")]
use rayon::prelude::*;

/// A simple new type wrapper around a `Vec` with the invariant that the length of the
/// inner `Vec` must not be greater than [`MAX_COLORS`].
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct Centroids<Color>(Vec<Color>);

impl<Color> Centroids<Color> {
    /// Unwrap into the inner `Vec<Color>` value.
    #[must_use]
    pub fn into_inner(self) -> Vec<Color> {
        self.0
    }

    /// Creates a [`Centroids`] without ensuring that its length
    /// is less than or equal to [`MAX_COLORS`].
    #[allow(unused)]
    pub(crate) const fn new_unchecked(centroids: Vec<Color>) -> Self {
        Self(centroids)
    }

    /// Creates a new [`Centroids`] by truncating the given `Vec` of colors to a max length of [`MAX_COLORS`].
    #[must_use]
    pub fn from_truncated(mut centroids: Vec<Color>) -> Self {
        centroids.truncate(MAX_COLORS.into());
        Self(centroids)
    }

    /// Returns the number of centroids/colors as a `u16`.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub fn num_colors(&self) -> u16 {
        self.0.len() as u16
    }

    /// Returns the number of centroids/colors as a [`PaletteSize`].
    #[must_use]
    pub fn palette_size(&self) -> PaletteSize {
        PaletteSize::new_unchecked(self.num_colors())
    }
}

impl<Color> AsRef<[Color]> for Centroids<Color> {
    fn as_ref(&self) -> &[Color] {
        self
    }
}

impl<Color> Deref for Centroids<Color> {
    type Target = [Color];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Color> From<Centroids<Color>> for Vec<Color> {
    fn from(centroids: Centroids<Color>) -> Self {
        centroids.into_inner()
    }
}

impl<Color> TryFrom<Vec<Color>> for Centroids<Color> {
    type Error = AboveMaxLen<u16>;

    fn try_from(colors: Vec<Color>) -> Result<Self, Self::Error> {
        if colors.len() <= usize::from(MAX_COLORS) {
            Ok(Self(colors))
        } else {
            Err(AboveMaxLen(MAX_COLORS))
        }
    }
}

/// Computes the index and lane of the point in `points` nearest to `query` according to euclidean distance.
#[inline]
fn simd_argmin<const N: usize>(points: &[[f32x8; N]], query: [f32; N]) -> (u8, u8) {
    let incr = u32x8::ONE;
    let mut cur_chunk = u32x8::ZERO;
    let mut min_chunk = cur_chunk;
    let mut min_distance = f32x8::splat(f32::INFINITY);

    let query = query.map(f32x8::splat);

    for chunk in points {
        #[allow(clippy::unwrap_used)]
        let distance = array::from_fn::<_, N, _>(|i| {
            let diff = query[i] - chunk[i];
            diff * diff
        })
        .into_iter()
        .reduce(|a, b| a + b)
        .unwrap();

        #[allow(unsafe_code)]
        let mask: u32x8 = unsafe { std::mem::transmute(distance.cmp_le(min_distance)) };
        min_chunk = mask.blend(cur_chunk, min_chunk);
        min_distance = min_distance.fast_min(distance);
        cur_chunk += incr;
    }

    let mut min_lane = 0;
    let mut min_dist = f32::INFINITY;
    for (i, &v) in min_distance.as_array_ref().iter().enumerate() {
        if v < min_dist {
            min_dist = v;
            min_lane = i;
        }
    }

    let min_chunk = min_chunk.as_array_ref()[min_lane];

    #[allow(clippy::cast_possible_truncation)]
    {
        (min_chunk as u8, min_lane as u8)
    }
}

/// The struct holding the data and state for k-means.
struct State<'a, Color, Component, const N: usize, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    ColorCount: ColorCounts<Color, Component, N>,
    f32: AsPrimitive<Component>,
{
    /// The component type must remain the same for each [`State`].
    _phantom: PhantomData<Component>,
    /// The color and count data.
    color_counts: &'a ColorCount,
    /// The components of the current centroids.
    components: Vec<[f32x8; N]>,
    /// The number of samples added to each centroid.
    counts: Vec<u32>,
    /// The output `Vec` for the final centroids/colors.
    output: Vec<Color>,
}

impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    ColorCount: ColorCounts<Color, Component, N>,
    f32: AsPrimitive<Component>,
{
    /// Create a new [`State`] with the given initial centroids.
    fn new(color_counts: &'a ColorCount, centroids: Vec<Color>) -> Self {
        let mut components = Vec::with_capacity(centroids.len().div_ceil(8));
        let chunks = centroids.as_arrays().chunks_exact(8);
        components.extend(
            chunks.clone().map(|chunk| {
                array::from_fn(|i| f32x8::new(array::from_fn(|j| chunk[j][i].into())))
            }),
        );

        if !chunks.remainder().is_empty() {
            let mut arr = [[f32::INFINITY; 8]; N];
            for (i, &color) in chunks.remainder().iter().enumerate() {
                for (arr, c) in arr.iter_mut().zip(color) {
                    arr[i] = c.into();
                }
            }
            components.push(arr.map(f32x8::new));
        }

        Self {
            _phantom: PhantomData,
            color_counts,
            components,
            counts: vec![0; centroids.len()],
            output: centroids,
        }
    }

    /// Adds the given color sample to its nearest centroid.
    #[inline]
    fn add_sample(&mut self, color: [Component; N]) {
        let Self { components, counts, .. } = self;
        let color = color.map(Into::into);

        let (chunk, lane) = simd_argmin(components, color);

        let chunk = usize::from(chunk);
        let lane = usize::from(lane);
        let i = chunk * 8 + lane;

        let count = counts[i] + 1;
        #[allow(clippy::cast_possible_truncation)]
        let rate = (1.0 / f64::from(count).sqrt()) as f32; // learning rate of 0.5 => count^(-0.5)

        let mut center = components[chunk].map(|v| v.as_array_ref()[lane]);
        for c in 0..N {
            center[c] += rate * (color[c] - center[c]);
        }

        for (d, s) in components[chunk].iter_mut().zip(center) {
            #[allow(unsafe_code)]
            let d = unsafe { &mut *(d as *mut f32x8).cast::<[f32; 8]>() };
            d[lane] = s;
        }

        counts[i] = count;
    }

    /// Runs online k-means using the given `distribution`.
    fn online_kmeans_inner(
        &mut self,
        samples: u32,
        seed: u64,
        distribution: &impl Distribution<usize>,
    ) {
        /// The number of samples to batch together.
        const BATCH: u32 = 256;

        let rng = &mut Xoroshiro128PlusPlus::seed_from_u64(seed);
        let colors = self.color_counts.color_components();

        let mut batch = Vec::with_capacity(BATCH as usize);

        for _ in 0..(samples / BATCH) {
            batch.extend((0..BATCH).map(|_| colors[distribution.sample(rng)]));

            for &color in &batch {
                self.add_sample(color);
            }

            batch.clear();
        }

        batch.extend((0..(samples % BATCH)).map(|_| colors[distribution.sample(rng)]));

        for color in batch {
            self.add_sample(color);
        }
    }

    /// Creates a new [`WeightedAliasIndex`] from the given `counts` data.
    #[allow(clippy::unwrap_used)]
    fn weighted_alias_index(counts: &[u32]) -> WeightedAliasIndex<u64> {
        // WeightedAliasIndex::new fails if:
        // - The vector is empty => should be handled by caller
        // - The vector is longer than u32::MAX =>
        //      ColorCounts implementors guarantee length <= MAX_PIXELS = u32::MAX
        // - For any weight w: w < 0 or w > max where max = W::MAX / weights.len() =>
        //      max count and max length are u32::MAX, so converting all counts to u64 will prevent this
        // - The sum of weights is zero =>
        //      ColorCounts implementors guarantee every count is > 0
        WeightedAliasIndex::new(counts.iter().copied().map(u64::from).collect()).unwrap()
    }

    /// Runs online k-means for the given number of samples.
    fn online_kmeans(&mut self, samples: u32, seed: u64) {
        if let Some(counts) = self.color_counts.counts() {
            let distribution = Self::weighted_alias_index(counts);
            self.online_kmeans_inner(samples, seed, &distribution);
        } else {
            let distribution = Uniform::new(0, self.color_counts.len());
            self.online_kmeans_inner(samples, seed, &distribution);
        }
    }

    /// Converts into a [`QuantizeOutput`], using the current centroids as the color palette.
    fn into_summary(self, indices: Vec<u8>) -> QuantizeOutput<Color> {
        let Self { components, counts, output, .. } = self;

        let mut palette = output;

        let len = palette.len();
        palette.clear();
        palette.extend(components.into_iter().flat_map(|x| {
            array::from_fn::<Color, 8, _>(|i| {
                cast::from_array(x.map(|y| y.as_array_ref()[i].as_()))
            })
        }));
        palette.truncate(len);

        QuantizeOutput { palette, counts, indices }
    }
}

impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    ColorCount: ColorCountsRemap<Color, Component, N>,
    f32: AsPrimitive<Component>,
{
    /// Computes the index of the nearest centroid for each color.
    fn indices(&mut self) -> Vec<u8> {
        self.color_counts
            .color_components()
            .iter()
            .map(|color| {
                let (chunk, lane) = simd_argmin(&self.components, color.map(Into::into));
                chunk * 8 + lane
            })
            .collect()
    }
}

/// Computes a color palette from the given `color_counts`.
///
/// The palette will have at most `initial_centroids.len()` number of colors.
///
/// See the [module](crate::kmeans) documentation for more information on the parameters.
#[must_use]
pub fn palette<Color, Component, const N: usize>(
    color_counts: &impl ColorCounts<Color, Component, N>,
    num_samples: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    f32: AsPrimitive<Component>,
{
    if initial_centroids.is_empty() || color_counts.is_empty() {
        QuantizeOutput::default()
    } else {
        let mut state = State::new(color_counts, initial_centroids.into());
        if num_samples > 0 {
            state.online_kmeans(num_samples, seed);
        }
        state.into_summary(Vec::new())
    }
}

/// Computes a color palette from the given `color_counts`.
/// The returned [`QuantizeOutput`] will have its `indices` populated.
///
/// The palette will have at most `initial_centroids.len()` number of colors.
///
/// See the [module](crate::kmeans) documentation for more information on the parameters.
#[must_use]
pub fn indexed_palette<Color, Component, const N: usize>(
    color_counts: &impl ColorCountsRemap<Color, Component, N>,
    num_samples: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static,
    f32: AsPrimitive<Component>,
{
    if initial_centroids.is_empty() || color_counts.is_empty() {
        QuantizeOutput::default()
    } else {
        let mut state = State::new(color_counts, initial_centroids.into());
        if num_samples > 0 {
            state.online_kmeans(num_samples, seed);
        }
        let indices = state.indices();
        state.into_summary(color_counts.map_indices(indices))
    }
}

#[cfg(feature = "threads")]
impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static + Send + Sync,
    ColorCount: ColorCounts<Color, Component, N>,
    f32: AsPrimitive<Component>,
{
    /// Runs minibatch k-means using the given `distribution`.
    fn minibatch_kmeans_inner(
        &mut self,
        max_samples: u32,
        batch_size: u32,
        seed: u64,
        distribution: &(impl Distribution<usize> + Sync),
    ) {
        /// Used to align to 64 bytes (most likely a cache line).
        // This is to prevent false sharing.
        // It doesn't seem to make a noticeable difference,
        // even though the multiple items in `rng` below would otherwise share the same cache line.
        // We'll keep this just in case, since it may make a difference on some hardware?
        #[repr(align(64))]
        struct Align64<T>(T);

        let Self { components, counts, color_counts, .. } = self;

        let colors = color_counts.color_components();

        let threads = rayon::current_num_threads();
        let chunk_size = (batch_size as usize).div_ceil(threads);

        let mut rng = (0..threads)
            .map(|i| Align64(Xoroshiro128PlusPlus::seed_from_u64(seed ^ i as u64)))
            .collect::<Vec<_>>();

        let mut batch = vec![[0.0.as_(); N]; batch_size as usize];
        let mut assignments = vec![(0, 0); batch_size as usize];

        for _ in 0..(max_samples / batch_size) {
            batch
                .par_chunks_mut(chunk_size)
                .zip(assignments.par_chunks_mut(chunk_size))
                .zip(&mut rng)
                .for_each(|((batch, assignments), Align64(rng))| {
                    for color in &mut *batch {
                        *color = colors[distribution.sample(rng)];
                    }

                    for (color, center) in batch.iter().zip(assignments) {
                        *center = simd_argmin(components, color.map(Into::into));
                    }
                });

            for (color, &(chunk, lane)) in batch.iter().zip(&assignments) {
                let color = color.map(Into::into);
                let chunk = usize::from(chunk);
                let lane = usize::from(lane);
                let i = chunk * 8 + lane;

                let count = counts[i] + 1;
                #[allow(clippy::cast_possible_truncation)]
                let rate = (1.0 / f64::from(count).sqrt()) as f32; // learning rate of 0.5 => count^(-0.5)

                #[allow(unsafe_code)]
                let centroid = unsafe {
                    &mut *std::ptr::addr_of_mut!(components[chunk]).cast::<[[f32; 8]; N]>()
                };
                for c in 0..N {
                    centroid[c][lane] += rate * (color[c] - centroid[c][lane]);
                }

                counts[i] = count;
            }
        }
    }

    /// Runs minibatch k-means for the given number of samples.
    fn minibatch_kmeans(&mut self, samples: u32, batch_size: u32, seed: u64) {
        if let Some(counts) = self.color_counts.counts() {
            let distribution = Self::weighted_alias_index(counts);
            self.minibatch_kmeans_inner(samples, batch_size, seed, &distribution);
        } else {
            let distribution = Uniform::new(0, self.color_counts.len());
            self.minibatch_kmeans_inner(samples, batch_size, seed, &distribution);
        }
    }
}

#[cfg(feature = "threads")]
impl<'a, Color, Component, const N: usize, ColorCount> State<'a, Color, Component, N, ColorCount>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static + Send + Sync,
    ColorCount: ColorCountsParallelRemap<Color, Component, N>,
    f32: AsPrimitive<Component>,
{
    /// Computes the index of the nearest centroid for each color in parallel.
    fn indices_par(&mut self) -> Vec<u8> {
        self.color_counts
            .color_components()
            .par_iter()
            .map(|color| {
                let (chunk, lane) = simd_argmin(&self.components, color.map(Into::into));
                chunk * 8 + lane
            })
            .collect()
    }
}

/// Computes a color palette in parallel from the given `color_counts`.
///
/// The palette will have at most `initial_centroids.len()` number of colors.
/// `floor(num_samples / batch_size)` batches will be run.
///
/// See the [module](crate::kmeans) documentation for more information on the parameters.
#[cfg(feature = "threads")]
#[must_use]
pub fn palette_par<Color, Component, const N: usize>(
    color_counts: &impl ColorCounts<Color, Component, N>,
    num_samples: u32,
    batch_size: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static + Send + Sync,
    f32: AsPrimitive<Component>,
{
    if initial_centroids.is_empty() || color_counts.is_empty() {
        QuantizeOutput::default()
    } else {
        let mut state = State::new(color_counts, initial_centroids.into());
        if num_samples >= batch_size && batch_size > 0 {
            state.minibatch_kmeans(num_samples, batch_size, seed);
        }
        state.into_summary(Vec::new())
    }
}

/// Computes a color palette in parallel from the given `color_counts`.
/// The returned [`QuantizeOutput`] will have its `indices` populated.
///
/// The palette will have at most `initial_centroids.len()` number of colors.
/// `floor(num_samples / batch_size)` batches will be run.
///
/// See the [module](crate::kmeans) documentation for more information on the parameters.
#[cfg(feature = "threads")]
#[must_use]
pub fn indexed_palette_par<Color, Component, const N: usize>(
    color_counts: &impl ColorCountsParallelRemap<Color, Component, N>,
    num_samples: u32,
    batch_size: u32,
    initial_centroids: Centroids<Color>,
    seed: u64,
) -> QuantizeOutput<Color>
where
    Color: ColorComponents<Component, N>,
    Component: Copy + Into<f32> + 'static + Send + Sync,
    f32: AsPrimitive<Component>,
{
    if initial_centroids.is_empty() || color_counts.is_empty() {
        QuantizeOutput::default()
    } else {
        let mut state = State::new(color_counts, initial_centroids.into());
        if num_samples >= batch_size && batch_size > 0 {
            state.minibatch_kmeans(num_samples, batch_size, seed);
        }
        let indices = state.indices_par();
        state.into_summary(color_counts.map_indices_par(indices))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    use crate::{tests::*, ColorSlice};

    use ordered_float::OrderedFloat;
    use palette::Srgb;

    fn test_centroids() -> Centroids<Srgb<u8>> {
        let mut centroids = test_data_256();
        centroids.truncate(249); // use non-multiple of 8 to test remainder handling
        Centroids::try_from(centroids).unwrap()
    }

    #[test]
    fn empty_inputs() {
        let colors = test_data_1024();
        let colors = &ColorSlice::try_from(colors.as_slice()).unwrap();
        let num_samples = 505;
        let centroids = test_centroids();
        let seed = 0;

        let empty_output = QuantizeOutput::default();

        let empty_colors = &ColorSlice::new_unchecked(&[]);
        assert_output_eq(
            &palette(empty_colors, num_samples, centroids.clone(), seed),
            &empty_output,
        );
        assert_output_eq(
            &indexed_palette(empty_colors, num_samples, centroids.clone(), seed),
            &empty_output,
        );
        #[cfg(feature = "threads")]
        {
            let batch_size = 64;

            assert_output_eq(
                &palette_par(
                    empty_colors,
                    num_samples,
                    batch_size,
                    centroids.clone(),
                    seed,
                ),
                &empty_output,
            );
            assert_output_eq(
                &indexed_palette_par(
                    empty_colors,
                    num_samples,
                    batch_size,
                    centroids.clone(),
                    seed,
                ),
                &empty_output,
            );
        }

        let empty_centroids = Centroids::new_unchecked(Vec::new());
        assert_output_eq(
            &palette(colors, num_samples, empty_centroids.clone(), seed),
            &empty_output,
        );
        assert_output_eq(
            &indexed_palette(colors, num_samples, empty_centroids.clone(), seed),
            &empty_output,
        );
        #[cfg(feature = "threads")]
        {
            let batch_size = 64;

            assert_output_eq(
                &palette_par(
                    colors,
                    num_samples,
                    batch_size,
                    empty_centroids.clone(),
                    seed,
                ),
                &empty_output,
            );
            assert_output_eq(
                &indexed_palette_par(
                    colors,
                    num_samples,
                    batch_size,
                    empty_centroids.clone(),
                    seed,
                ),
                &empty_output,
            );
        }
    }

    #[test]
    fn no_samples_gives_initial_centroids() {
        let colors = test_data_1024();
        let colors = &ColorSlice::try_from(colors.as_slice()).unwrap();
        let num_samples = 0;
        let centroids = test_centroids();
        let seed = 0;

        let actual = palette(colors, num_samples, centroids.clone(), seed);
        let expected = QuantizeOutput {
            palette: centroids.to_vec(),
            counts: vec![0; centroids.len()],
            indices: Vec::new(),
        };
        assert_output_eq(&actual, &expected);

        let actual = indexed_palette(colors, num_samples, centroids.clone(), seed);
        assert_eq!(actual.indices.len(), colors.len());
        let expected_indexed = QuantizeOutput {
            indices: actual.indices.clone(),
            ..expected.clone()
        };
        assert_output_eq(&actual, &expected_indexed);

        #[cfg(feature = "threads")]
        {
            let num_samples = 0;
            let batch_size = 64;
            let actual = palette_par(colors, num_samples, batch_size, centroids.clone(), seed);
            assert_output_eq(&actual, &expected);
            let actual =
                indexed_palette_par(colors, num_samples, batch_size, centroids.clone(), seed);
            assert_eq!(actual.indices.len(), colors.len());
            let expected_indexed = QuantizeOutput {
                indices: actual.indices.clone(),
                ..expected.clone()
            };
            assert_output_eq(&actual, &expected_indexed);

            let num_samples = 505;
            let batch_size = 0;
            let actual = palette_par(colors, num_samples, batch_size, centroids.clone(), seed);
            assert_output_eq(&actual, &expected);
            let actual =
                indexed_palette_par(colors, num_samples, batch_size, centroids.clone(), seed);
            assert_eq!(actual.indices.len(), colors.len());
            let expected_indexed = QuantizeOutput {
                indices: actual.indices.clone(),
                ..expected.clone()
            };
            assert_output_eq(&actual, &expected_indexed);
        }
    }

    #[test]
    fn exact_match_image_unaffected() {
        let centroids = test_centroids();

        let indices = {
            #[allow(clippy::cast_possible_truncation)]
            let indices = (0..centroids.len()).map(|i| i as u8).collect::<Vec<_>>();
            let mut indices = [indices.as_slice(); 4].concat();
            indices.rotate_right(7);
            indices
        };

        let colors = indices
            .iter()
            .map(|&i| centroids[usize::from(i)])
            .collect::<Vec<_>>();

        let colors = &ColorSlice::try_from(colors.as_slice()).unwrap();

        let num_samples = 505;
        let seed = 0;

        let actual = palette(colors, num_samples, centroids.clone(), seed);
        assert_eq!(actual.counts.len(), centroids.len());
        let expected = QuantizeOutput {
            palette: centroids.to_vec(),
            counts: actual.counts.clone(),
            indices: Vec::new(),
        };
        assert_output_eq(&actual, &expected);
        assert_eq!(actual.counts.into_iter().sum::<u32>(), num_samples);

        let actual = indexed_palette(colors, num_samples, centroids.clone(), seed);
        assert_eq!(actual.counts.len(), centroids.len());
        let expected = QuantizeOutput {
            palette: centroids.to_vec(),
            counts: actual.counts.clone(),
            indices: indices.clone(),
        };
        assert_output_eq(&actual, &expected);
        assert_eq!(actual.counts.into_iter().sum::<u32>(), num_samples);

        #[cfg(feature = "threads")]
        {
            let batch_size = 64;

            let actual = palette_par(colors, num_samples, batch_size, centroids.clone(), seed);
            assert_eq!(actual.counts.len(), centroids.len());
            let expected = QuantizeOutput {
                palette: centroids.to_vec(),
                counts: actual.counts.clone(),
                indices: Vec::new(),
            };
            assert_output_eq(&actual, &expected);
            assert_eq!(
                actual.counts.into_iter().sum::<u32>(),
                num_samples - num_samples % batch_size
            );

            let actual =
                indexed_palette_par(colors, num_samples, batch_size, centroids.clone(), seed);
            assert_eq!(actual.counts.len(), centroids.len());
            let expected = QuantizeOutput {
                palette: centroids.to_vec(),
                counts: actual.counts.clone(),
                indices,
            };
            assert_output_eq(&actual, &expected);
            assert_eq!(
                actual.counts.into_iter().sum::<u32>(),
                num_samples - num_samples % batch_size
            );
        }
    }

    #[test]
    fn naive_nearest_neighbor_oracle() {
        fn squared_euclidean_distance<const N: usize>(x: [f32; N], y: [f32; N]) -> f32 {
            let mut dist = 0.0;
            for c in 0..N {
                let d = x[c] - y[c];
                dist += d * d;
            }
            dist
        }

        let centroids = to_float_arrays(&test_centroids());
        let points = to_float_arrays(&test_data_1024());

        let mut components = Vec::with_capacity(centroids.len().div_ceil(8));
        let chunks = centroids.chunks_exact(8);
        components.extend(
            chunks
                .clone()
                .map(|chunk| array::from_fn(|i| f32x8::new(array::from_fn(|j| chunk[j][i])))),
        );

        if !chunks.remainder().is_empty() {
            let mut arr = [[f32::INFINITY; 8]; 3];
            for (i, &color) in chunks.remainder().iter().enumerate() {
                for (arr, c) in arr.iter_mut().zip(color) {
                    arr[i] = c;
                }
            }
            components.push(arr.map(f32x8::new));
        }

        for color in points {
            #[allow(clippy::unwrap_used)]
            let expected = centroids
                .iter()
                .map(|&centroid| OrderedFloat(squared_euclidean_distance(centroid, color)))
                .min()
                .unwrap()
                .0;

            let (chunk, lane) = simd_argmin(&components, color);
            let index = usize::from(chunk) * 8 + usize::from(lane);
            let actual = squared_euclidean_distance(color, centroids[index]);

            #[allow(clippy::float_cmp)]
            {
                assert_eq!(expected, actual);
            }
        }
    }
}
