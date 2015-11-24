{
  "targets": [
    {
      "target_name": "draw",
      "sources": [ "draw.cc" ],
      "cflags_cc": [ "-fexceptions" ],
      "include_dirs": [
        "<!(node -e \"require('nan')\")",
        "/Users/tpeterka/software/mfa/install/include",
        "/Users/tpeterka/software/diy2/include",
        "/Users/tpeterka/software/eigen-3.2.5"
      ],
      "libraries" : ["/Users/tpeterka/software/mfa/install/lib/libmfa.dylib"]
    }
  ]
}
