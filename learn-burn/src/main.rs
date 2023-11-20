use crate::mnist::MNISTBatcher;
use crate::model::ModelConfig;
use crate::training::TrainingConfig;
use burn::autodiff::ADBackendDecorator;
use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::WgpuBackend;
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::source::huggingface::MNISTItem;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use burn::tensor::backend::Backend;

pub mod mnist;
pub mod model;
pub mod training;

use std::path::Path;

fn main() {
    type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp";
    crate::training::train::<MyAutodiffBackend>(
        artifact_dir,
        crate::training::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MNISTItem) {
    let config =
        TrainingConfig::load(format!("{artifact_dir}/config.json")).expect("A config exists");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Failed to load trained model");

    let model = config.model.init_with::<B>(record).to_device(&device);

    let label = item.label;
    let batcher = MNISTBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}
