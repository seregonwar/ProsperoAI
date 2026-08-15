pai_define_library(pai_gpu_kernels
  vecadd_code.c
)

target_include_directories(pai_gpu_kernels PUBLIC
  ${PROJECT_SOURCE_DIR}/gpu/kernels
)

# ---------------------------------------------------------------------
# Shader assembly rule (gfx1013): regenerate the .inc from the .s when
# llvm-mc is available; otherwise use the checked-in prebuilt blob.
# ---------------------------------------------------------------------

set(PAI_VECADD_S ${CMAKE_CURRENT_SOURCE_DIR}/gfx1013/vecadd.s)
set(PAI_VECADD_INC ${CMAKE_CURRENT_SOURCE_DIR}/gfx1013/pai_vecadd_code.inc)

if(PAI_ASSEMBLE_SHADERS)
  find_program(PAI_LLVM_MC llvm-mc)
  find_program(PAI_LLVM_OBJCOPY llvm-objcopy)

  # On Windows the llvm tools usually live inside WSL.
  if(NOT PAI_LLVM_MC AND WIN32)
    find_program(PAI_WSL wsl)
    if(PAI_WSL)
      execute_process(COMMAND ${PAI_WSL} which llvm-mc
        OUTPUT_VARIABLE PAI_LLVM_MC_WSL OUTPUT_STRIP_TRAILING_WHITESPACE)
      execute_process(COMMAND ${PAI_WSL} which llvm-objcopy
        OUTPUT_VARIABLE PAI_LLVM_OBJCOPY_WSL OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(PAI_LLVM_MC_WSL)
        set(PAI_LLVM_MC_FOUND TRUE)
      endif()
    endif()
  else()
    set(PAI_LLVM_MC_FOUND TRUE)
  endif()

  if(PAI_LLVM_MC_FOUND)
    if(WIN32 AND NOT PAI_LLVM_MC)
      # Run the whole assembly inside WSL with converted paths.
      function(pai_path_to_wsl in_path out_var)
        string(REPLACE "\\" "/" _p "${in_path}")
        if(_p MATCHES "^([A-Za-z]):(.*)$")
          string(TOLOWER "${CMAKE_MATCH_1}" _drive)
          set(_p "/mnt/${_drive}${CMAKE_MATCH_2}")
        endif()
        set(${out_var} "${_p}" PARENT_SCOPE)
      endfunction()

      pai_path_to_wsl("${PAI_VECADD_S}" PAI_VECADD_S_WSL)
      pai_path_to_wsl("${PAI_VECADD_INC}" PAI_VECADD_INC_WSL)
      pai_path_to_wsl("${PROJECT_SOURCE_DIR}/toolchain/assemble_shader.py"
        PAI_SCRIPT_WSL)

      set(PAI_ASSEMBLE_CMD ${PAI_WSL} python3 ${PAI_SCRIPT_WSL}
        ${PAI_LLVM_MC_WSL} ${PAI_LLVM_OBJCOPY_WSL}
        ${PAI_VECADD_S_WSL} ${PAI_VECADD_INC_WSL})
    else()
      find_program(PAI_PYTHON python)
      if(NOT PAI_PYTHON)
        set(PAI_LLVM_MC_FOUND FALSE)
      else()
        set(PAI_ASSEMBLE_CMD ${PAI_PYTHON}
          ${PROJECT_SOURCE_DIR}/toolchain/assemble_shader.py
          ${PAI_LLVM_MC} ${PAI_LLVM_OBJCOPY} ${PAI_VECADD_S} ${PAI_VECADD_INC})
      endif()
    endif()
  endif()

  if(PAI_LLVM_MC_FOUND)
    # Assemble into the source tree so the checked-in .inc stays in sync.
    add_custom_command(
      OUTPUT ${PAI_VECADD_INC}
      COMMAND ${PAI_ASSEMBLE_CMD}
      DEPENDS ${PAI_VECADD_S} ${PROJECT_SOURCE_DIR}/toolchain/assemble_shader.py
      COMMENT "Assembling gfx1013 shader: vecadd"
      VERBATIM
    )
    add_custom_target(pai-assemble-shaders DEPENDS ${PAI_VECADD_INC})
    add_dependencies(pai_gpu_kernels pai-assemble-shaders)
  else()
    message(WARNING
      "llvm-mc/python not found; using checked-in prebuilt shader blob")
  endif()
endif()
