{
  "targets": [
    {
      "target_name": "draw",
      "sources": [ "draw.cc" ],
      "include_dirs": [
        "<!(node -e \"require('nan')\")",
        "/Users/tpeterka/software/mfa/install/include",
        "/Users/tpeterka/software/diy/include",
        "/usr/local/include/eigen3"
      ],
      "libraries" : ["/Users/tpeterka/software/mfa/install/lib/libmfa.dylib"]
    }
  ]
}
