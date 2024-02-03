//! Contains the [`UnsizedPipeline`] builder struct for the high level API.

#[cfg(feature = "kmeans")]
use super::num_samples;

use crate::{
    wu::{self, Binner3},
    ColorComponents, ColorCountsRemap, ColorSlice, ColorSpace, PalettePipeline,
    PaletteSize, QuantizeMethod, SumPromotion, ZeroedIsZero,
};

#[cfg(all(feature = "colorspaces", feature = "threads"))]
use crate::colorspace::convert_color_slice_par;
#[cfg(feature = "colorspaces")]
use crate::colorspace::{convert_color_slice, from_srgb, to_srgb};
#[cfg(feature = "image")]
use crate::AboveMaxLen;
#[cfg(feature = "threads")]
use crate::ColorCountsParallelRemap;
#[cfg(any(feature = "colorspaces", feature = "kmeans"))]
use crate::IndexedColorCounts;
#[cfg(feature = "kmeans")]
use crate::{
    kmeans::{self, Centroids},
    KmeansOptions,
};

use num_traits::AsPrimitive;
use palette::Srgb;

#[cfg(feature = "image")]
use image::RgbImage;
#[cfg(feature = "colorspaces")]
use palette::{Lab, Oklab};

/// A builder struct to specify options to create a quantized image or an indexed palette from an image.
///
/// # Examples
/// To start, create a [`UnsizedPipeline`] from a [`RgbImage`] (note that the `image` feature is needed):
/// ```no_run
/// # use quantette::UnsizedPipeline;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let img = image::open("some image")?.into_rgb8();
/// let pipeline = ImagePipeline::try_from(&img)?;
/// # Ok(())
/// # }
/// ```
///
/// Then, you can change different options like the number of colors in the palette:
/// ```
/// # use quantette::{ImagePipeline, AboveMaxLen, ColorSpace, QuantizeMethod, KmeansOptions};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let pipeline = ImagePipeline::new(srgb.as_slice().try_into()?, 1, 1).unwrap();
/// let pipeline = pipeline
///     .palette_size(192.into())
///     .colorspace(ColorSpace::Oklab)
///     .quantize_method(QuantizeMethod::kmeans());
/// # Ok(())
/// # }
/// ```
///
/// [`ImagePipeline`] has all options that [`PalettePipeline`] does,
/// so you can check its documentation example for more information.
/// In addition, [`ImagePipeline`] has options to control the dither behavior:
/// ```
/// # use quantette::{ImagePipeline, AboveMaxLen, ColorSpace, QuantizeMethod, KmeansOptions};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let pipeline = ImagePipeline::new(srgb.as_slice().try_into()?, 1, 1).unwrap();
/// let pipeline = pipeline
///     .palette_size(192.into())
///     .colorspace(ColorSpace::Oklab)
///     .quantize_method(QuantizeMethod::kmeans())
///     .dither_error_diffusion(0.8);
/// # Ok(())
/// # }
/// ```
///
/// Finally, run the pipeline:
/// ```no_run
/// # use quantette::{ImagePipeline, AboveMaxLen};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let pipeline = ImagePipeline::new(srgb.as_slice().try_into()?, 1, 1).unwrap();
/// let image = pipeline.quantized_rgbimage();
/// # Ok(())
/// # }
/// ```
///
/// Or, in parallel across multiple threads (needs the `threads` feature):
/// ```no_run
/// # use quantette::{ImagePipeline, AboveMaxLen};
/// # use palette::Srgb;
/// # fn main() -> Result<(), AboveMaxLen<u32>> {
/// # let srgb = vec![Srgb::new(0, 0, 0)];
/// # let pipeline = ImagePipeline::new(srgb.as_slice().try_into()?, 1, 1).unwrap();
/// let image = pipeline.quantized_rgbimage_par();
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct UnsizedPipeline<'a> {
    /// The input image as a flat slice of pixels.
    pub(crate) colors: ColorSlice<'a, Srgb<u8>>,
    /// The number of colors to put in the palette.
    pub(crate) k: PaletteSize,
    /// The color space to perform color quantization in.
    pub(crate) colorspace: ColorSpace,
    /// The color quantization method to use.
    pub(crate) quantize_method: QuantizeMethod<Srgb<u8>>,
    /// Whether or not to deduplicate the input pixels/colors.
    #[cfg(any(feature = "kmeans", feature = "colorspaces"))]
    pub(crate) dedup_pixels: bool,
}

impl<'a> UnsizedPipeline<'a> {
    /// Creates a new [`UnsizedPipeline`] with default options
    pub fn new(colors: ColorSlice<'a, Srgb<u8>>) -> Self {
        Self {
            colors,
            k: PaletteSize::default(),
            colorspace: ColorSpace::Srgb,
            quantize_method: QuantizeMethod::wu(),
            #[cfg(any(feature = "kmeans", feature = "colorspaces"))]
            dedup_pixels: true,
        }
    }

    /// Sets the palette size which determines the (maximum) number of colors to have in the palette.
    ///
    /// The default palette size is [`PaletteSize::MAX`].
    #[must_use]
    pub fn palette_size(mut self, size: PaletteSize) -> Self {
        self.k = size;
        self
    }

    /// Sets the color space to perform color quantization in.
    ///
    /// See [`ColorSpace`] for more details.
    ///
    /// The default color space is [`ColorSpace::Srgb`].
    #[must_use]
    #[cfg(feature = "colorspaces")]
    pub fn colorspace(mut self, colorspace: ColorSpace) -> Self {
        self.colorspace = colorspace;
        self
    }

    /// Sets the color quantization method to use.
    ///
    /// See [`QuantizeMethod`] for more details.
    ///
    /// The default quantization method is [`QuantizeMethod::Wu`].
    #[must_use]
    #[cfg(feature = "kmeans")]
    pub fn quantize_method(mut self, quantize_method: QuantizeMethod<Srgb<u8>>) -> Self {
        self.quantize_method = quantize_method;
        self
    }

    /// Sets whether or not to deduplicate pixels in the image.
    ///
    /// It is recommended to keep this option as default, unless the image is very small or
    /// you have reason to believe that the image contains very little redundancy
    /// (i.e., most pixels are their own unique color).
    /// `quantette` will only deduplicate pixels if it is worth doing so
    /// (i.e., the k-means quantization method is chosen or a color space converision is needed).
    ///
    /// The default value is `true`.
    #[must_use]
    #[cfg(any(feature = "colorspaces", feature = "kmeans"))]
    pub fn dedup_pixels(mut self, dedup_pixels: bool) -> Self {
        self.dedup_pixels = dedup_pixels;
        self
    }
}

#[cfg(feature = "image")]
impl<'a> TryFrom<&'a RgbImage> for UnsizedPipeline<'a> {
    type Error = AboveMaxLen<u32>;

    fn try_from(image: &'a RgbImage) -> Result<Self, Self::Error> {
        Ok(Self::new(image.try_into()?))
    }
}

impl<'a> UnsizedPipeline<'a> {
    /// Runs the pipeline and returns the computed color palette.
    #[must_use]
    pub fn palette(self) -> Vec<Srgb<u8>> {
        PalettePipeline::from(self).palette()
    }

    /// Runs the pipeline and returns the quantized image as an indexed palette.
    #[must_use]
    pub fn indexed_palette(self) -> (Vec<Srgb<u8>>, Vec<u8>) {
        match self.colorspace {
            ColorSpace::Srgb => {
                let Self {
                    colors,
                    k,
                    quantize_method,
                    #[cfg(feature = "kmeans")]
                    dedup_pixels,
                    ..
                } = self;

                let binner = ColorSpace::default_binner_srgb_u8();

                match quantize_method {
                    #[cfg(feature = "kmeans")]
                    QuantizeMethod::Kmeans(options) if dedup_pixels => {
                        let color_counts = IndexedColorCounts::new(colors, |c| c);
                        indexed_palette(
                            &color_counts,
                            k,
                            QuantizeMethod::Kmeans(options),
                            &binner,
                        )
                    }
                    quantize_method => indexed_palette(
                        &colors,
                        k,
                        quantize_method,
                        &binner,
                    ),
                }
            }
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => self.indexed_palette_convert(
                &ColorSpace::default_binner_lab_f32(),
                from_srgb::<Lab>,
                to_srgb,
            ),
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => self.indexed_palette_convert(
                &ColorSpace::default_binner_oklab_f32(),
                from_srgb::<Oklab>,
                to_srgb,
            ),
        }
    }

    /// Computes an indexed palette, converting to a different color space to perform the quantization.
    #[cfg(feature = "colorspaces")]
    fn indexed_palette_convert<Color, Component, const B: usize>(
        self,
        binner: &impl Binner3<Component, B>,
        convert_to: impl Fn(Srgb<u8>) -> Color,
        convert_back: impl Fn(Color) -> Srgb<u8>,
    ) -> (Vec<Srgb<u8>>, Vec<u8>)
    where
        Color: ColorComponents<Component, 3>,
        Component: SumPromotion<u32> + Into<f32>,
        Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
        u32: Into<Component::Sum>,
        f32: AsPrimitive<Component>,
    {
        let Self {
            colors,
            k,
            quantize_method,
            dedup_pixels,
            ..
        } = self;

        let quantize_method = quantize_method.convert_color_space_from_srgb(&convert_to);

        let (palette, indices) = if dedup_pixels {
            let color_counts = IndexedColorCounts::new(colors, convert_to);
            indexed_palette(
                &color_counts,
                k,
                quantize_method,
                binner,
            )
        } else {
            let colors = convert_color_slice(colors, convert_to);
            let colors = ColorSlice::new_unchecked(&colors);
            indexed_palette(&colors, k, quantize_method, binner)
        };

        let palette = palette.into_iter().map(convert_back).collect();
        (palette, indices)
    }
}

#[cfg(feature = "threads")]
impl<'a> UnsizedPipeline<'a> {
    /// Runs the pipeline in parallel and returns the computed color palette.
    #[must_use]
    pub fn palette_par(self) -> Vec<Srgb<u8>> {
        PalettePipeline::from(self).palette_par()
    }

    /// Runs the pipeline in parallel and returns the quantized image as an indexed palette.
    #[must_use]
    pub fn indexed_palette_par(self) -> (Vec<Srgb<u8>>, Vec<u8>) {
        match self.colorspace {
            ColorSpace::Srgb => {
                let Self {
                    colors,
                    k,
                    quantize_method,
                    #[cfg(feature = "kmeans")]
                    dedup_pixels,
                    ..
                } = self;

                let binner = ColorSpace::default_binner_srgb_u8();

                match quantize_method {
                    #[cfg(feature = "kmeans")]
                    QuantizeMethod::Kmeans(options) if dedup_pixels => {
                        let color_counts = IndexedColorCounts::new_par(colors, |c| c);
                        indexed_palette_par(
                            &color_counts,
                            k,
                            QuantizeMethod::Kmeans(options),
                            &binner,
                        )
                    }
                    quantize_method => indexed_palette_par(
                        &colors,
                        k,
                        quantize_method,
                        &binner,
                    ),
                }
            }
            #[cfg(feature = "colorspaces")]
            ColorSpace::Lab => self.indexed_palette_convert_par(
                &ColorSpace::default_binner_lab_f32(),
                from_srgb::<Lab>,
                to_srgb,
            ),
            #[cfg(feature = "colorspaces")]
            ColorSpace::Oklab => self.indexed_palette_convert_par(
                &ColorSpace::default_binner_oklab_f32(),
                from_srgb::<Oklab>,
                to_srgb,
            ),
        }
    }

    /// Computes an indexed palette in parallel, converting to a different color space
    /// to perform the quantization.
    #[cfg(feature = "colorspaces")]
    fn indexed_palette_convert_par<Color, Component, const B: usize>(
        self,
        binner: &(impl Binner3<Component, B> + Sync),
        convert_to: impl Fn(Srgb<u8>) -> Color + Send + Sync,
        convert_back: impl Fn(Color) -> Srgb<u8>,
    ) -> (Vec<Srgb<u8>>, Vec<u8>)
    where
        Color: ColorComponents<Component, 3> + Send + Sync,
        Component: SumPromotion<u32> + Into<f32> + Send + Sync,
        Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
        u32: Into<Component::Sum>,
        f32: AsPrimitive<Component>,
    {
        let Self {
            colors,
            k,
            quantize_method,
            dedup_pixels,
            ..
        } = self;

        let quantize_method = quantize_method.convert_color_space_from_srgb(&convert_to);

        let (palette, indices) = if dedup_pixels {
            let color_counts = IndexedColorCounts::new_par(colors, convert_to);
            indexed_palette_par(
                &color_counts,
                k,
                quantize_method,
                binner,
            )
        } else {
            let colors = convert_color_slice_par(colors, convert_to);
            let colors = ColorSlice::new_unchecked(&colors);
            indexed_palette_par(&colors, k, quantize_method, binner)
        };

        let palette = palette.into_iter().map(convert_back).collect();
        (palette, indices)
    }
}

/// Computes a color palette and the indices into it.
#[allow(clippy::needless_pass_by_value)]
fn indexed_palette<Color, Component, const B: usize>(
    color_counts: &impl ColorCountsRemap<Color, Component, 3>,
    k: PaletteSize,
    method: QuantizeMethod<Color>,
    binner: &impl Binner3<Component, B>,
) -> (Vec<Color>, Vec<u8>)
where
    Color: ColorComponents<Component, 3>,
    Component: SumPromotion<u32> + Into<f32>,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64>,
    u32: Into<Component::Sum>,
    f32: AsPrimitive<Component>,
{
    match method {
        QuantizeMethod::Wu(_) => {
            let res = wu::indexed_palette(color_counts, k, binner);
            (res.palette, res.indices)
        }
        #[cfg(feature = "kmeans")]
        QuantizeMethod::Kmeans(KmeansOptions {
            sampling_factor, initial_centroids, seed, ..
        }) => {
            let initial_centroids = initial_centroids.unwrap_or_else(|| {
                Centroids::new_unchecked(wu::palette(color_counts, k, binner).palette)
            });
            let num_samples = num_samples(sampling_factor, color_counts);
            let res = kmeans::indexed_palette(color_counts, num_samples, initial_centroids, seed);
            (res.palette, res.indices)
        }
    }
}

/// Computes a color palette and the indices into it in parallel.
#[cfg(feature = "threads")]
#[allow(clippy::needless_pass_by_value)]
fn indexed_palette_par<Color, Component, const B: usize>(
    color_counts: &(impl ColorCountsParallelRemap<Color, Component, 3> + Send + Sync),
    k: PaletteSize,
    method: QuantizeMethod<Color>,
    binner: &(impl Binner3<Component, B> + Sync),
) -> (Vec<Color>, Vec<u8>)
where
    Color: ColorComponents<Component, 3> + Send,
    Component: SumPromotion<u32> + Into<f32> + Send + Sync,
    Component::Sum: ZeroedIsZero + AsPrimitive<f64> + Send,
    u32: Into<Component::Sum>,
    f32: AsPrimitive<Component>,
{
    match method {
        QuantizeMethod::Wu(_) => {
            let res = wu::indexed_palette_par(color_counts, k, binner);
            (res.palette, res.indices)
        }
        #[cfg(feature = "kmeans")]
        QuantizeMethod::Kmeans(KmeansOptions {
            sampling_factor,
            initial_centroids,
            seed,
            batch_size,
        }) => {
            let initial_centroids = initial_centroids.unwrap_or_else(|| {
                Centroids::new_unchecked(wu::palette_par(color_counts, k, binner).palette)
            });
            let num_samples = num_samples(sampling_factor, color_counts);
            let res = kmeans::indexed_palette_par(
                color_counts,
                num_samples,
                batch_size,
                initial_centroids,
                seed,
            );
            (res.palette, res.indices)
        }
    }
}
