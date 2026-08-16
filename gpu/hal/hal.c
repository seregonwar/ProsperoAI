#include <hal/hal.h>

#include <pai/log.h>

#include <stdlib.h>
#include <string.h>

#ifdef PAI_PS5
extern const pai_gpu_backend_ops_t pai_gpu_ops_ps5_gc;
#endif
extern const pai_gpu_backend_ops_t pai_gpu_ops_host_ref;

pai_status_t
pai_gpu_device_create(pai_gpu_backend_t backend, pai_gpu_device_t **out_device) {
  pai_gpu_device_t *dev;
  pai_status_t st;

  if (!out_device) {
    return PAI_ERR_INVALID_ARG;
  }

  dev = (pai_gpu_device_t *)calloc(1, sizeof(*dev));
  if (!dev) {
    return PAI_ERR_NOMEM;
  }

  switch (backend) {
#ifdef PAI_PS5
  case PAI_GPU_BACKEND_PS5_GC:
    dev->ops = &pai_gpu_ops_ps5_gc;
    dev->backend = PAI_GPU_BACKEND_PS5_GC;
    break;
#endif
  case PAI_GPU_BACKEND_HOST_REF:
    dev->ops = &pai_gpu_ops_host_ref;
    dev->backend = PAI_GPU_BACKEND_HOST_REF;
    break;
  default:
    free(dev);
    return PAI_ERR_UNSUPPORTED;
  }

  st = dev->ops->init(dev);
  if (st != PAI_OK) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "backend '%s' init failed (%s)\n",
                   dev->ops->name, pai_status_str(st));
    free(dev);
    return st;
  }

  PAI_LOG_INFO_(PAI_SUB_GPU, "device created: %s\n", dev->ops->name);
  *out_device = dev;
  return PAI_OK;
}

uint64_t
pai_gpu_aux_va(pai_gpu_device_t *device) {
  if (!device || !device->ops || !device->ops->aux_va) {
    return 0ULL;
  }
  return device->ops->aux_va(device);
}

void
pai_gpu_device_destroy(pai_gpu_device_t *device) {
  if (!device) {
    return;
  }
  device->ops->shutdown(device);
  free(device);
}

const char *
pai_gpu_device_name(const pai_gpu_device_t *device) {
  return device->ops->name;
}

pai_gpu_backend_t
pai_gpu_device_backend(const pai_gpu_device_t *device) {
  return device->backend;
}

pai_status_t
pai_gpu_buffer_alloc(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer,
                     uint64_t size, uint32_t flags) {
  memset(buffer, 0, sizeof(*buffer));
  return device->ops->buffer_alloc(device, buffer, size, flags);
}

void
pai_gpu_buffer_free(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer) {
  device->ops->buffer_free(device, buffer);
}

pai_status_t
pai_gpu_submit(pai_gpu_device_t *device, const uint32_t *pm4, uint32_t dwords) {
  return device->ops->submit(device, pm4, dwords, PAI_GPU_QUEUE_DEFAULT);
}

pai_status_t
pai_gpu_submit_q(pai_gpu_device_t *device, const uint32_t *pm4, uint32_t dwords,
                 uint32_t queue_type) {
  return device->ops->submit(device, pm4, dwords, queue_type);
}

pai_status_t
pai_gpu_submit_acb(pai_gpu_device_t *device, const uint32_t *pm4,
                   uint32_t dwords) {
  /* ACB path: queue type 0xc = the special compute queue, and the CB
   * descriptor uses the const-IB header (0x33). Implemented via the
   * submit op with the queue_type's high bit marking const-IB mode. */
  return device->ops->submit(device, pm4, dwords, 0x8000000Cu);
}

pai_status_t
pai_gpu_wait_label(pai_gpu_device_t *device, uint64_t label_addr,
                   uint32_t label_value, uint64_t timeout_ns) {
  return device->ops->wait_label(device, label_addr, label_value, timeout_ns);
}

pai_status_t
pai_gpu_reset(pai_gpu_device_t *device) {
  return device->ops->reset(device);
}

pai_status_t
pai_gpu_submit_wait(pai_gpu_device_t *device, const uint32_t *pm4,
                    uint32_t dwords, uint64_t label_addr, uint32_t label_value,
                    uint64_t timeout_ns) {
  pai_status_t st = device->ops->submit(device, pm4, dwords,
                                        PAI_GPU_QUEUE_DEFAULT);
  if (st != PAI_OK) {
    return st;
  }
  return device->ops->wait_label(device, label_addr, label_value, timeout_ns);
}
