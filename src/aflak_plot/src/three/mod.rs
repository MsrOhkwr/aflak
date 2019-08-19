//! Draw 3D representations.

use std::borrow::Cow;

use glium::{
    backend::Facade,
    texture::{ClientFormat, RawImage2d},
    Texture2d,
};
use imgui::{ImTexture, Ui};
use ndarray::{ArrayBase, Data, Ix3, IxDyn};

extern crate xorshift;
use self::xorshift::{Rng, SeedableRng, Xorshift128};

use super::imshow::Textures;

/// TODO
pub trait UiImage3d {
    fn image3d<S, F>(
        &self,
        image: &ArrayBase<S, IxDyn>,
        texture_id: ImTexture,
        textures: &mut Textures,
        ctx: &F,
    ) where
        S: Data<Elem = f32>,
        F: Facade;
}

impl<'ui> UiImage3d for Ui<'ui> {
    // TODO
    fn image3d<S, F>(
        &self,
        image: &ArrayBase<S, IxDyn>,
        texture_id: ImTexture,
        textures: &mut Textures,
        ctx: &F,
    ) where
        S: Data<Elem = f32>,
        F: Facade,
    {
        let p = self.get_cursor_screen_pos();
        let window_pos = self.get_window_pos();
        let window_size = self.get_window_size();
        let size = (
            window_size.0 - 15.0,
            window_size.1 - (p.1 - window_pos.1) - 10.0,
        );

        // 3D image...
        let raw = make_raw_image(image);
        let gl_texture = Texture2d::new(ctx, raw).expect("Error!");
        textures.replace(texture_id, gl_texture);

        self.image(texture_id, size).build();
    }
}

struct Ray {
    origin: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>,
    direction: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>,
    color: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>,
}

impl Ray {
    pub fn new(x: f32, y: f32) -> Ray {
        let o = array![0f32, 0f32, -2.0f32];
        let mut dir = array![x, y, -1.5f32] - &o;
        let len = dir.dot(&dir).sqrt();
        dir = dir / len;
        Ray {
            origin: o,
            direction: dir,
            color: array![0f32, 0f32, 0f32],
        }
    }
}

fn in_bounding_volume(ray: &mut Ray) -> bool {
    let n_dim = 3;
    let min = array![-1f32, -1f32, -1f32];
    let max = array![1f32, 1f32, 1f32];
    let mut tmin = 0f32;
    let mut tmax = std::f32::MAX;
    let mut result = true;

    for i in 0..n_dim {
        let t0 = ((min[i] - ray.origin[i]) / ray.direction[i])
            .min((max[i] - ray.origin[i]) / ray.direction[i]);
        let t1 = ((min[i] - ray.origin[i]) / ray.direction[i])
            .max((max[i] - ray.origin[i]) / ray.direction[i]);
        tmin = t0.max(tmin);
        tmax = t1.min(tmax);
        if tmax < tmin {
            result = false;
            break;
        }
    }

    if result {
        ray.origin = &ray.origin + &(&ray.direction * tmin);
    }

    result
}

fn color_legend(val: f32) -> (ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>) {
    let temp = (1f32 - (4f32 * val * std::f32::consts::PI).cos()) / 2f32;
    if val > 1f32 {
        array![1f32, 0f32, 0f32]
    } else if val > 3f32 / 4.0f32 {
        array![1f32, temp, 0f32] * val
    } else if val > 2f32 / 4.0f32 {
        array![temp, 1f32, 0f32] * val
    } else if val > 1f32 / 4.0f32 {
        array![0f32, 1f32, temp] * val
    } else if val > 0f32 {
        array![0f32, temp, 1f32] * val
    } else {
        array![0f32, 0f32, 0f32]
    }
}

fn ray_casting<S>(image: &ArrayBase<S, Ix3>, n: usize, m: usize, spp: usize) -> Vec<u8>
where
    S: Data<Elem = f32>,
{
    let states = [123456789, 987654321];
    let mut rng: Xorshift128 = SeedableRng::from_seed(&states[..]);

    let shape = image.dim();
    let mut data = Vec::with_capacity(3 * n * m);

    for u in 0..n {
        for v in 0..m {
            let mut color = array![0f32, 0f32, 0f32];

            for sample in 1..(spp + 1usize) {
                let mut ray = Ray::new(
                    ((u as f32 + rng.next_f32()) / n as f32 - 0.5f32) * 2f32,
                    ((v as f32 + rng.next_f32()) / m as f32 - 0.5f32) * 2f32,
                );

                let dt = 1f32 / (shape.0.max(shape.1)).max(shape.2) as f32;
                let mut step = 1;
                while in_bounding_volume(&mut ray) {
                    let val = image[[
                        ((ray.origin[0] + 1f32) / 2f32 * (shape.0 as f32 - 0.001f32)) as usize,
                        ((ray.origin[1] + 1f32) / 2f32 * (shape.1 as f32 - 0.001f32)) as usize,
                        ((ray.origin[2] + 1f32) / 2f32 * (shape.2 as f32 - 0.001f32)) as usize,
                    ]];
                    let temp = (color_legend(val) - &ray.color) / step as f32;
                    ray.color = ray.color + temp;
                    step += 1;
                    ray.origin = ray.origin + &ray.direction * dt;
                }

                let temp = (ray.color - &color) / sample as f32;
                color = color + temp;
            }
            data.push((color[0] * 255.999f32) as u8);
            data.push((color[1] * 255.999f32) as u8);
            data.push((color[2] * 255.999f32) as u8);
        }
    }

    data
}

fn make_raw_image<S>(image: &ArrayBase<S, IxDyn>) -> RawImage2d<'static, u8>
where
    S: Data<Elem = f32>,
{
    let image3 = image.slice(s![.., .., ..]);
    let n = 64;
    let m = 64;
    let spp = 8;
    let data = ray_casting(&image3, n, m, spp);
    RawImage2d {
        data: Cow::Owned(data),
        width: n as u32,
        height: m as u32,
        format: ClientFormat::U8U8U8,
    }
}
