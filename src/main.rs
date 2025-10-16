use std::sync::Arc;

use anyhow::{anyhow, Result};
use glam::{Mat4, Vec2, Vec3};
use instant::Instant;
use vulkano::buffer::{
    allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator},
    Buffer, BufferContents, BufferCreateInfo, BufferUsage,
};
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator,
    AutoCommandBufferBuilder, CommandBufferUsage,
};
use vulkano::descriptor_set::{
    allocator::StandardDescriptorSetAllocator,
    PersistentDescriptorSet, WriteDescriptorSet,
};
use vulkano::device::{
    physical::PhysicalDevice,
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
};
use vulkano::format::Format;
use vulkano::image::{
    view::ImageView,
    AttachmentImage, ImageDimensions, ImageUsage, SwapchainImage,
};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::{
    graphics::{
        depth_stencil::DepthStencilState,
        input_assembly::InputAssemblyState,
        vertex_input::Vertex,
        viewport::{Viewport, ViewportState},
    },
    GraphicsPipeline, Pipeline, PipelineBindPoint,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    acquire_next_image, AcquireError, PresentMode, Surface, Swapchain, SwapchainCreateInfo,
    SwapchainPresentInfo,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::VulkanLibrary;
use vulkano_win::{create_surface_from_winit, required_extensions};
use winit::event::{ElementState, Event, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

#[derive(BufferContents, Vertex, Clone, Copy, Default)]
#[repr(C)]
struct VertexData {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct UniformBufferObject {
    view_proj: [[f32; 4]; 4],
}

struct WaterSimulation {
    width: usize,
    height: usize,
    heights: Vec<f32>,
    previous: Vec<f32>,
    damping: f32,
}

impl WaterSimulation {
    fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            width,
            height,
            heights: vec![0.0; size],
            previous: vec![0.0; size],
            damping: 0.995,
        }
    }

    fn disturb(&mut self, x: usize, y: usize, magnitude: f32) {
        if x < self.width && y < self.height {
            let idx = y * self.width + x;
            self.heights[idx] = magnitude;
        }
    }

    fn step(&mut self) {
        let mut next = self.heights.clone();
        for y in 1..self.height - 1 {
            for x in 1..self.width - 1 {
                let idx = y * self.width + x;
                let north = self.heights[(y - 1) * self.width + x];
                let south = self.heights[(y + 1) * self.width + x];
                let east = self.heights[y * self.width + (x + 1)];
                let west = self.heights[y * self.width + (x - 1)];
                let current = self.heights[idx];
                let previous = self.previous[idx];
                let wave = (north + south + east + west) * 0.25 - current;
                let new_height = wave + (current - previous) * self.damping;
                next[idx] = new_height;
            }
        }
        self.previous.copy_from_slice(&self.heights);
        self.heights.copy_from_slice(&next);
    }

    fn vertices(&self) -> Vec<VertexData> {
        let mut vertices = Vec::with_capacity(self.width * self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let height = self.heights[y * self.width + x];
                let fx = x as f32 / (self.width - 1) as f32;
                let fy = y as f32 / (self.height - 1) as f32;
                let position = [
                    fx * 2.0 - 1.0,
                    height,
                    fy * 2.0 - 1.0,
                ];
                vertices.push(VertexData {
                    position,
                    uv: [fx, fy],
                });
            }
        }
        vertices
    }

    fn indices(&self) -> Vec<u32> {
        let mut indices = Vec::with_capacity((self.width - 1) * (self.height - 1) * 6);
        for y in 0..self.height - 1 {
            for x in 0..self.width - 1 {
                let i0 = (y * self.width + x) as u32;
                let i1 = (y * self.width + x + 1) as u32;
                let i2 = ((y + 1) * self.width + x) as u32;
                let i3 = ((y + 1) * self.width + x + 1) as u32;
                indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
            }
        }
        indices
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec3 frag_pos;
layout(location = 1) out vec2 frag_uv;
layout(set = 0, binding = 0) uniform Data {
    mat4 view_proj;
} uniforms;
void main() {
    frag_pos = position;
    frag_uv = uv;
    gl_Position = uniforms.view_proj * vec4(position.x, position.y, position.z, 1.0);
}",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "#version 450
layout(location = 0) in vec3 frag_pos;
layout(location = 1) in vec2 frag_uv;
layout(location = 0) out vec4 f_color;
void main() {
    float intensity = clamp(0.5 + frag_pos.y * 0.8, 0.1, 1.0);
    vec3 water_color = mix(vec3(0.0, 0.1, 0.6), vec3(0.1, 0.4, 0.8), intensity);
    f_color = vec4(water_color, 1.0);
}",
    }
}

struct Camera {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

impl Camera {
    fn view_projection(&self, aspect: f32) -> Mat4 {
        let center = Vec3::ZERO;
        let yaw_rot = Mat4::from_rotation_y(self.yaw);
        let pitch_rot = Mat4::from_rotation_x(self.pitch);
        let direction = (yaw_rot * pitch_rot).transform_vector3(Vec3::new(0.0, -0.5, 1.5)).normalize();
        let eye = center - direction * self.distance;
        let view = Mat4::look_at_rh(eye, center, Vec3::Y);
        let proj = Mat4::perspective_rh_gl(45f32.to_radians(), aspect, 0.1, 100.0);
        proj * view
    }
}

fn main() -> Result<()> {
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new().with_title("Vulkan Water Simulation").build(&event_loop)?);

    let library = VulkanLibrary::new()?;
    let required_extensions = required_extensions(window.as_ref());
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )?;
    let surface = create_surface_from_winit(window.clone(), instance.clone())?;

    let (physical_device, queue_family_index) = select_device(&instance, &surface)?;
    let (device, queue) = create_device(&physical_device, queue_family_index)?;

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let command_buffer_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());

    let mut swapchain_state = create_swapchain(device.clone(), surface.clone(), window.clone())?;

    let mut simulation = WaterSimulation::new(64, 64);
    simulation.disturb(32, 32, 0.2);
    let indices = simulation.indices();
    let index_buffer = Buffer::from_iter(
        memory_allocator.as_ref(),
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        indices.clone().into_iter(),
    )?;

    let vs = vs::load(device.clone())?;
    let fs = fs::load(device.clone())?;

    let render_pass = swapchain_state.render_pass.clone();
    let mut pipeline = create_pipeline(
        device.clone(),
        vs,
        fs,
        render_pass.clone(),
        swapchain_state.image_extent,
    );

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut camera = Camera {
        yaw: 0.0,
        pitch: -0.4,
        distance: 3.0,
    };
    let mut mouse_pressed = false;
    let mut last_cursor = Vec2::ZERO;
    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(_) => recreate_swapchain = true,
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        mouse_pressed = state == ElementState::Pressed;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let pos = Vec2::new(position.x as f32, position.y as f32);
                    if mouse_pressed {
                        let delta = pos - last_cursor;
                        camera.yaw += delta.x * 0.005;
                        camera.pitch = (camera.pitch + delta.y * 0.005).clamp(-1.2, 0.2);
                    }
                    last_cursor = pos;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(key) = input.virtual_keycode {
                        if input.state == ElementState::Pressed {
                            match key {
                                VirtualKeyCode::W | VirtualKeyCode::Z => {
                                    camera.distance = (camera.distance - 0.1).max(1.5);
                                }
                                VirtualKeyCode::S | VirtualKeyCode::X => {
                                    camera.distance = (camera.distance + 0.1).min(6.0);
                                }
                                VirtualKeyCode::Space => {
                                    simulation.disturb(32, 32, 0.3);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    recreate_swapchain = false;
                    if let Err(err) = recreate_swapchain_state(
                        device.clone(),
                        window.clone(),
                        &mut swapchain_state,
                        &mut pipeline,
                    ) {
                        eprintln!("Failed to recreate swapchain: {err}");
                        return;
                    }
                }

                let now = Instant::now();
                let _delta = now - last_frame;
                last_frame = now;

                simulation.step();

                let vertices = simulation.vertices();
                let vertex_buffer = match Buffer::from_iter(
                    memory_allocator.as_ref(),
                    BufferCreateInfo {
                        usage: BufferUsage::VERTEX_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        usage: MemoryUsage::Upload,
                        ..Default::default()
                    },
                    vertices.into_iter(),
                ) {
                    Ok(buffer) => buffer,
                    Err(err) => {
                        eprintln!("Failed to create vertex buffer: {err}");
                        return;
                    }
                };

                let (image_index, suboptimal, acquire_future) = match acquire_next_image(swapchain_state.swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(err) => panic!("failed to acquire image: {err}"),
                };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let aspect = {
                    let size = window.inner_size();
                    size.width as f32 / size.height as f32
                };
                let view_proj = camera.view_projection(aspect);
                let uniform_buffer = match Buffer::from_data(
                    memory_allocator.as_ref(),
                    BufferCreateInfo {
                        usage: BufferUsage::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    AllocationCreateInfo {
                        usage: MemoryUsage::Upload,
                        ..Default::default()
                    },
                    UniformBufferObject {
                        view_proj: view_proj.to_cols_array_2d(),
                    },
                ) {
                    Ok(buffer) => buffer,
                    Err(err) => {
                        eprintln!("Failed to create uniform buffer: {err}");
                        return;
                    }
                };

                let layout = pipeline.layout().set_layouts().get(0).unwrap().clone();
                let set = PersistentDescriptorSet::new(
                    &descriptor_set_allocator,
                    layout,
                    [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
                )
                .unwrap();

                let mut builder = match AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                ) {
                    Ok(b) => b,
                    Err(err) => {
                        eprintln!("Failed to create command buffer: {err}");
                        return;
                    }
                };

                let viewport = Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [
                        swapchain_state.image_extent[0] as f32,
                        swapchain_state.image_extent[1] as f32,
                    ],
                    depth_range: 0.0..1.0,
                };

                builder
                    .begin_render_pass(
                        swapchain_state.framebuffers[image_index as usize].clone(),
                        vulkano::command_buffer::SubpassContents::Inline,
                        vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()],
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()])
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        set,
                    )
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .bind_index_buffer(index_buffer.clone())
                    .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();

                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(
                        queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            swapchain_state.swapchain.clone(),
                            image_index,
                        ),
                    )
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        eprintln!("Failed to flush future: {e}");
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            }
            _ => {}
        }
    });
}

struct SwapchainState {
    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    image_format: Format,
    image_extent: [u32; 2],
    framebuffers: Vec<Arc<Framebuffer>>,
    depth_image: Arc<AttachmentImage>,
    render_pass: Arc<RenderPass>,
}

fn create_swapchain(
    device: Arc<Device>,
    surface: Arc<Surface<Window>>,
    window: Arc<Window>,
) -> Result<SwapchainState> {
    let physical_device = device.physical_device();
    let capabilities = physical_device
        .surface_capabilities(&surface, Default::default())?;
    let formats = physical_device.surface_formats(&surface, Default::default())?;

    let image_format = formats
        .iter()
        .find(|(format, _)| *format == Format::B8G8R8A8_SRGB)
        .map(|(format, _)| *format)
        .unwrap_or(formats[0].0);

    let image_extent = window.inner_size();
    let image_extent = [image_extent.width, image_extent.height];

    let (swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: capabilities.min_image_count + 1,
            image_format: Some(image_format),
            image_extent,
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: capabilities.supported_composite_alpha.iter().next().unwrap(),
            present_mode: PresentMode::Fifo,
            ..Default::default()
        },
    )?;

    let image_format = swapchain.image_format();

    let render_pass = create_render_pass(device.clone(), image_format);

    let depth_image = AttachmentImage::with_usage(
        device.clone(),
        ImageDimensions::Dim2d {
            width: image_extent[0],
            height: image_extent[1],
            array_layers: 1,
        },
        Format::D16_UNORM,
        ImageUsage::DEPTH_STENCIL_ATTACHMENT,
    )?;

    let framebuffers = create_framebuffers(&images, depth_image.clone(), render_pass.clone());

    Ok(SwapchainState {
        swapchain,
        images,
        image_format,
        image_extent,
        framebuffers,
        depth_image,
        render_pass,
    })
}

fn create_render_pass(device: Arc<Device>, color_format: Format) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: color_format,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap()
}

fn create_framebuffers(
    images: &[Arc<SwapchainImage<Window>>],
    depth_image: Arc<AttachmentImage>,
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            let depth_view = ImageView::new_default(depth_image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect()
}

fn create_pipeline(
    device: Arc<Device>,
    vs: vs::Shader,
    fs: fs::Shader,
    render_pass: Arc<RenderPass>,
    extent: [u32; 2],
) -> Arc<GraphicsPipeline> {
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [extent[0] as f32, extent[1] as f32],
        depth_range: 0.0..1.0,
    };

    GraphicsPipeline::start()
        .vertex_input_state(VertexData::per_vertex())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([viewport]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass, 0).unwrap())
        .build(device.clone())
        .unwrap()
}

fn recreate_swapchain_state(
    device: Arc<Device>,
    window: Arc<Window>,
    swapchain_state: &mut SwapchainState,
    pipeline: &mut Arc<GraphicsPipeline>,
) -> Result<()> {
    let new_size = window.inner_size();
    if new_size.width == 0 || new_size.height == 0 {
        return Ok(());
    }
    let (new_swapchain, new_images) = match swapchain_state.swapchain.recreate(SwapchainCreateInfo {
        image_extent: [new_size.width, new_size.height],
        ..swapchain_state.swapchain.create_info()
    }) {
        Ok(r) => r,
        Err(vulkano::swapchain::SwapchainCreationError::ImageExtentNotSupported { .. }) => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    swapchain_state.swapchain = new_swapchain;
    swapchain_state.images = new_images;
    swapchain_state.image_extent = [new_size.width, new_size.height];
    swapchain_state.image_format = swapchain_state.swapchain.image_format();
    swapchain_state.depth_image = AttachmentImage::with_usage(
        device.clone(),
        ImageDimensions::Dim2d {
            width: swapchain_state.image_extent[0],
            height: swapchain_state.image_extent[1],
            array_layers: 1,
        },
        Format::D16_UNORM,
        ImageUsage::DEPTH_STENCIL_ATTACHMENT,
    )?;
    swapchain_state.render_pass = create_render_pass(device.clone(), swapchain_state.image_format);
    swapchain_state.framebuffers = create_framebuffers(
        &swapchain_state.images,
        swapchain_state.depth_image.clone(),
        swapchain_state.render_pass.clone(),
    );
    *pipeline = create_pipeline(
        device.clone(),
        vs::load(device.clone())?,
        fs::load(device.clone())?,
        swapchain_state.render_pass.clone(),
        swapchain_state.image_extent,
    );
    Ok(())
}

fn select_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface<Window>>,
) -> Result<(Arc<PhysicalDevice>, u32)> {
    instance
        .enumerate_physical_devices()?
        .filter_map(|device| {
            let queue_family_index = device
                .queue_family_properties()
                .iter()
                .enumerate()
                .find(|(index, info)| {
                    info.queue_flags.graphics
                        && device
                            .surface_support(*index as u32, surface)
                            .unwrap_or(false)
                })
                .map(|(index, _)| index as u32)?;
            Some((device, queue_family_index))
        })
        .next()
        .ok_or_else(|| anyhow!("No suitable physical device found"))
}

fn create_device(
    physical_device: &Arc<PhysicalDevice>,
    queue_family_index: u32,
) -> Result<(Arc<Device>, Arc<Queue>)> {
    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..DeviceExtensions::default()
            },
            ..Default::default()
        },
    )?;
    let queue = queues.next().ok_or_else(|| anyhow!("Missing queue"))?;
    Ok((device, queue))
}
