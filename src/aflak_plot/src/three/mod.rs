//! Draw 3D representations.

use std::borrow::Cow;
use glium::{
    backend::Facade,
    texture::{ClientFormat, RawImage2d},
    Texture2d,
    Surface,
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

#[derive(Copy, Clone)]
struct Vertex {
    pos: [f32; 3],
    texcoord: [f32; 2],
}
implement_vertex!(Vertex, pos, texcoord);

fn ray_casting_gpu<S>(image: &ArrayBase<S, Ix3>, n: usize, m: usize) -> Vec<u8>
where
    S: Data<Elem = f32>,
{
    let mut data = vec![0; 3 * n * m];

    let events_loop = glium::glutin::EventsLoop::new();
    let wb = glium::glutin::WindowBuilder::new()
        .with_visibility(false);
    let cb = glium::glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &events_loop).unwrap();
    let program = glium::Program::from_source(display.get_context(),
        "#version 430

        in vec3 pos;
        in vec2 texcoord;
        out vec2 Texcoord;
        void main() {
          gl_Position = vec4(pos, 1.0);
          Texcoord = texcoord;
        }",
        "#version 430

        #define PI (3.14159265359)
        #define FLT_MAX (3.402823466e+38)

        in vec2 Texcoord;
        out vec4 out_color;
        uniform sampler3D volume;

        uvec4 xors;

        float rand() {
            const uint t = (xors[0] ^ (xors[0] << 11));
            xors[0] = xors[1];
            xors[1] = xors[2];
            xors[2] = xors[3];
            xors[3] = (xors[3] ^ (xors[3] >> 19)) ^ (t ^ (t >> 18));
            return xors[3] / 4294967295.0f;
        }

        struct ray {
            vec3 origin;
            vec3 direction;
            vec4 color;
        };

        bool hit_volume(inout ray r) {
            const int nDim = 3;
            const vec3 _min = {-1.0f, -1.0f, -1.0f};
            const vec3 _max = {1.0f, 1.0f, 1.0f};
            float tmin = 0.0f;
            float tmax = FLT_MAX;
            float t0;
            float t1;
            for (int i = 0; i < nDim; i++) {
                t0 = min((_min[i] - r.origin[i]) / r.direction[i],
                        (_max[i] - r.origin[i]) / r.direction[i]);
                t1 = max((_min[i] - r.origin[i]) / r.direction[i],
                        (_max[i] - r.origin[i]) / r.direction[i]);
                tmin = max(t0, tmin);
                tmax = min(t1, tmax);
                if (tmax <= tmin) {
                        return false;
                }
            }
            r.origin = r.origin + tmin * r.direction;
            return true;
        }

        vec4 color_legend(const in float val) {
            const float temp = (-cos(4.0f * val * PI) + 1.0f) / 2.0f;
            const vec4 result =
                (val > 1.0f) ? vec4(1.0f, 0.0f, 0.0f, 0.0f) :
                (val > 3.0f / 4.0f) ? vec4(1.0f, temp, 0.0f, 0.0f) * val :
                (val > 2.0f / 4.0f) ? vec4(temp, 1.0f, 0.0f, 0.0f) * val :
                (val > 1.0f / 4.0f) ? vec4(0.0f, 1.0f, temp, 0.0f) * val :
                (val > 0.0f) ? vec4(0.0f, temp, 1.0f, 0.0f) * val : vec4(0.0f, 0.0f, 0.0f, 0.0f);

            return result;
        }

        vec4 toneMap(const in vec4 color, const in float white) {
            return clamp(color * (1 + color / white) / (1 + color), 0, 1);
        }

        void mat_volume(inout ray r) {
            const float dt = 1.0f / 10.0f;
            uint step = 1;
            float val;
            while (hit_volume(r))
            {
                val = texture(volume, r.origin / 2.0f + vec3(0.5)).r;
                r.color += (color_legend(val) - r.color) / step++;
                r.origin += r.direction * dt;
            }
        }

        vec4 gammaCorrect(const in vec4 color, const in float gamma) {
            const float g = 1.0f / gamma;
            const vec4 result =
            {
                pow(color.r, g),
                pow(color.g, g),
                pow(color.b, g),
                1.0f,
            };
            return result;
        }

        void main() {
            const vec3 eye = vec3(0.0f, 0.0f, -2.0f);
            const vec3 position_screen =
            {
                Texcoord.x * 2.0f - 1.0f,
                Texcoord.y * 2.0f - 1.0f,
                eye.z + 0.5f,
            };

            ray r =
            {
                eye,
                normalize(position_screen - eye),
                vec4(0.0f),
            };

            mat_volume(r);

            //out_color = vec4(Texcoord, 0, 0);
            out_color = r.color;
        }",
        None
    ).unwrap();

    let fb_tex = Texture2d::empty_with_format(display.get_context(), glium::texture::UncompressedFloatFormat::F32F32F32F32, glium::texture::MipmapsOption::NoMipmap, n as u32, m as u32).unwrap();

    let mut fb = glium::framebuffer::SimpleFrameBuffer::new(display.get_context(), &fb_tex).unwrap();

    let vertex_buffer = glium::VertexBuffer::new(display.get_context(), &[
        Vertex{pos: [1.0, -1.0, 0.0], texcoord: [1.0, 1.0]},
        Vertex{pos: [-1.0, -1.0, 0.0], texcoord: [0.0, 1.0]},
        Vertex{pos: [-1.0, 1.0, 0.0], texcoord: [0.0, 0.0]},
        Vertex{pos: [1.0, 1.0, 0.0], texcoord: [1.0, 0.0]},
    ]).unwrap();
    let index_buffer = glium::index::NoIndices(glium::index::PrimitiveType::TriangleFan);
    let shape = image.dim();
    let mut volume_data = vec![vec![vec![0f32; shape.2]; shape.1]; shape.0];
    let mut min_val = std::f32::MAX;
    let mut max_val = 0f32;
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                if min_val > image[[i, j, k]] {
                    min_val = image[[i, j, k]];
                }
                if max_val < image[[i, j, k]] {
                    max_val = image[[i, j, k]];
                }
            }
        }
    }
    for i in 0..shape.0 {
        for j in 0..shape.1 {
            for k in 0..shape.2 {
                volume_data[i][j][k] = (if image[[i, j, k]] < 0f32 { 0f32 } else { image[[i, j, k]] } - min_val) / (max_val - min_val);
            }
        }
    }
    let texture = glium::texture::CompressedTexture3d::new(display.get_context(), glium::texture::Texture3dDataSource::into_raw(volume_data)).unwrap();
    let uniforms = uniform!{volume: texture.sampled().magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)};
    fb.draw(&vertex_buffer, &index_buffer, &program, &uniforms, &Default::default()).unwrap();

    let read_back: Vec<Vec<(u8, u8, u8, u8)>> = fb_tex.read();

    for i in 0..n {
        for j in 0..m {
            data[(i * m + j) * 3 + 0] = read_back[i][j].0;
            data[(i * m + j) * 3 + 1] = read_back[i][j].1;
            data[(i * m + j) * 3 + 2] = read_back[i][j].2;
        }
    }

    data
}

fn make_raw_image<S>(image: &ArrayBase<S, IxDyn>) -> RawImage2d<'static, u8>
where
    S: Data<Elem = f32>,
{
    let image3 = image.slice(s![.., .., ..]);
    let n = 512;
    let m = 512;
    let data = ray_casting_gpu(&image3, n, m);
    RawImage2d {
        data: Cow::Owned(data),
        width: n as u32,
        height: m as u32,
        format: ClientFormat::U8U8U8,
    }
}
