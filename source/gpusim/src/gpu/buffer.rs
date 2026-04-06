//! GPU buffer readback utility.

use crate::error::GpuSimError;

/// Maps a staging buffer, polls the device, reads raw bytes, and unmaps.
///
/// The caller must have already submitted a `copy_buffer_to_buffer` command
/// targeting `staging_buffer` before calling this function.
///
/// Returns the raw bytes as a `Vec<u8>`. Use `bytemuck::cast_slice` to
/// interpret as typed data.
pub(crate) fn readback_staging_buffer(
    device: &wgpu::Device,
    staging_buffer: &wgpu::Buffer,
    size: u64,
) -> Result<Vec<u8>, GpuSimError> {
    let buffer_slice = staging_buffer.slice(..size);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuSimError::DevicePollFailed(format!("{e}")))?;
    receiver
        .recv()
        .map_err(|_| GpuSimError::ChannelDisconnected)?
        .map_err(|e| GpuSimError::BufferMapRejected(format!("{e}")))?;

    let data = buffer_slice.get_mapped_range();
    let result = data.to_vec();
    drop(data);
    staging_buffer.unmap();

    Ok(result)
}
