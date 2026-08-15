# Regenerate gfx1013 shader blobs from their .s sources.
#
# Usage: python assemble_shader.py <llvm-mc> <llvm-objcopy> <input.s> <output.inc>
#
# Emits a C include file with uint32_t words, compatible with the
# kernel headers in gpu/kernels/.

import struct
import subprocess
import sys
import tempfile
import os


def main():
    if len(sys.argv) != 5:
        print("usage: assemble_shader.py <llvm-mc> <llvm-objcopy> <in.s> <out.inc>")
        return 1

    llvm_mc, llvm_objcopy, src, out = sys.argv[1:]

    with tempfile.TemporaryDirectory() as tmp:
        obj = os.path.join(tmp, "kernel.o")
        bin_ = os.path.join(tmp, "kernel.bin")

        r = subprocess.run(
            [llvm_mc, "-triple=amdgcn", "-mcpu=gfx1013",
             "-filetype=obj", src, "-o", obj])
        if r.returncode != 0:
            print("llvm-mc failed", file=sys.stderr)
            return r.returncode

        r = subprocess.run(
            [llvm_objcopy, "--dump-section", f".text={bin_}", obj])
        if r.returncode != 0:
            print("llvm-objcopy failed", file=sys.stderr)
            return r.returncode

        data = open(bin_, "rb").read()
        if len(data) % 4 != 0:
            print("shader size is not dword aligned", file=sys.stderr)
            return 1

        words = [struct.unpack("<I", data[i:i + 4])[0]
                 for i in range(0, len(data), 4)]

    lines = ["/* Generated from {} by llvm-mc (gfx1013). Do not edit. */".format(
        os.path.basename(src))]
    for i in range(0, len(words), 4):
        chunk = ", ".join("0x%08Xu" % w for w in words[i:i + 4])
        lines.append("    %s," % chunk)

    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("wrote %s (%d words)" % (out, len(words)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
