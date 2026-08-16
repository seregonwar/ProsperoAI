# ProsperoAI — OpenAI-compatible gateway (whitepaper §26)
#
# Host-side network layer: JSON + HTTP/1.1 + the OpenAI REST surface.
# Not compiled into PS5 payload builds (the payload talks the Prospero
# Protocol; the gateway is the desktop/local-endpoint side of §26).
if(PAI_PLATFORM STREQUAL "host")
  pai_define_library(pai_gateway json.c gwsys.c http.c gateway.c)
  target_link_libraries(pai_gateway PRIVATE pai_models)
  if(WIN32)
    target_link_libraries(pai_gateway PRIVATE ws2_32)
  else()
    find_package(Threads REQUIRED)
    target_link_libraries(pai_gateway PRIVATE Threads::Threads)
  endif()
endif()
