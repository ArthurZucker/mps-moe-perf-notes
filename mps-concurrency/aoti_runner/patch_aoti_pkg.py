import os, re, subprocess, sys, zipfile
pkg, out = sys.argv[1], sys.argv[2]
work = os.path.dirname(os.path.abspath(out)) + '/_aoti_work'
os.makedirs(work, exist_ok=True)
with zipfile.ZipFile(pkg) as z:
    names = z.namelist()
    cpp_name = next(n for n in names if n.endswith('.wrapper.cpp'))
    so_name = next(n for n in names if n.endswith('.wrapper.so'))
    s = z.read(cpp_name).decode()
    z.extractall(work)
# patch source
if 'mps_concurrent_begin' not in s:
    s = s.replace('#include <vector>\n', '#include <vector>\n\nextern "C" void mps_concurrent_begin();\nextern "C" void mps_concurrent_finish();\n', 1)
    first = s.find('aoti_torch_mps_run_command_block(')
    last = s.rfind('aoti_torch_mps_run_command_block(')
    assert first != -1 and last != -1 and first != last
    le = s.find('\n', last) + 1
    s = s[:le] + '    mps_concurrent_finish();\n' + s[le:]
    ls = s.rfind('\n', 0, first) + 1
    s = s[:ls] + '    mps_concurrent_begin();\n' + s[ls:]
    open(os.path.join(work, cpp_name), 'w').write(s)
# compile
torch_dir = subprocess.check_output([sys.executable, '-c', 'import torch,os;print(os.path.dirname(torch.__file__))'], text=True).strip()
inc = subprocess.check_output([sys.executable, '-c', 'import sysconfig;print("-I"+sysconfig.get_paths()["include"])'], text=True).strip()
cmd = ['clang++','-shared','-fPIC','-std=c++20','-undefined','dynamic_lookup','-O2','-DNDEBUG','-DUSE_MPS',
       '-DTORCH_INDUCTOR_CPP_WRAPPER','-DSTANDALONE_TORCH_HEADER','-DTORCH_INDUCTOR_PRECOMPILE_HEADERS',
       '-DC10_USING_CUSTOM_GENERATED_MACROS','-DCPU_CAPABILITY_NEON','-DAT_BUILD_ARM_VEC256_WITH_SLEEF',
       os.path.join(work,cpp_name), os.path.join(os.path.dirname(os.path.abspath(__file__)),'concurrent_runner.mm'),
       inc, f'-I{torch_dir}/include', f'-I{torch_dir}/include/torch/csrc/api/include',
       f'-L{torch_dir}/lib', '-lomp', '-framework','Metal','-framework','Foundation','-framework','QuartzCore',
       f'-Wl,-rpath,{torch_dir}/lib', '-o', os.path.join(work,'custom.so')]
subprocess.run(cmd, check=True)
# repackage
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as zout:
    for item in zipfile.ZipFile(pkg).infolist():
        data = zipfile.ZipFile(pkg).read(item.filename)
        if item.filename == so_name:
            data = open(os.path.join(work,'custom.so'),'rb').read()
        zout.writestr(item,data)
print('wrote',out)
