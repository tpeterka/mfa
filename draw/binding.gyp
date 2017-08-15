{
  "targets": [
    {
      "target_name": "draw",
      "sources": [ "draw.cc" ],
      "include_dirs": [
        "<!(node -e \"require('nan')\")",
        "/homes/iulian/lib/mfa/include",
        "/homes/iulian/source/diy/include",
        "/homes/iulian/3rdparty/eigen/include/eigen3",
        "/homes/fathom/3rdparty/mpich/3.1/gnu/include"
      ],
      "libraries" : [
                       "/homes/iulian/lib/mfa/lib/libmfa.so", 
                       "/homes/fathom/3rdparty/mpich/3.1/gnu/lib/libmpich.so"
                    ]
    }
  ]
}
