# draw

Draw is a webGL renderer to draw geometry contained in a diy2 output
file produced by MFA

The key features are:

- C++, DIY, MPI server using a node.js addon reads the data
- Combined client and server in a standalone app using node webkit (nwjs)
- Javascript / webGL rendering on the client (using three.js rendering library instead of webGL directly)

The steps to build and run the example are below.

# Install dependencies:

- [node.js](https://nodejs.org/)
- [nwjs](https://github.com/nwjs/nw.js/)
- [three.js](http://threejs.org/) (for 3d rendering, a library wrapping webGL) Version 71 is bundled in the ```3rdparty``` directory.
- [dat.gui](https://code.google.com/p/dat-gui/) (for GUI controls) Version 0.5 is bundled in the ```3rdparty``` directory.
- nan
    - ```cd draw; npm install nan```
- bindings
    - ```cd draw; npm install bindings```
- nw-gyp
    - ```cd draw; npm install -g nw-gyp```

# To create a node.js addon:

See [Node.js addon instructions](https://github.com/nodejs/node-addon-examples)

- Add or edit existing package.json file
- Add or edit existing binding.gyp file
- Add or edit a .cc or .cxx file for the node addon C/C++ server code
- Add or edit one or more.js files for the javascript client code
- Add or edit index.html file

# To build the executable:

This is a standalone nwjs application, which is an app that looks and feels like a browser.

```cd draw```

Edit the path names in binding.gyp to your own. Then,

- To build an nwjs (node webkit) program requires compiling with
  nw-gyp and also needs to know the current version of nwjs. A few
  other C++11 flags are needed as follows
```
CC=mpicc \
CXX=mpicxx \
CXXFLAGS='-fexceptions -DBUILD_GYP -std=c++11 -stdlib=libc++ -mmacosx-version-min=10.9' \
LDFLAGS='-mmacosx-version-min=10.9' \
nw-gyp \
rebuild \
--target=0.19.5     # version number of nw.js (not node)
```
- The above command is in ```make/make-nwjs```.

# To run the program:

Generate a test input file named ```approx.out``` by running one of
the MFA examples. The file path in nwjs-client.js is hard-coded to
mfa/install/examples/nd/approx.out. If that is not correct, edit the
path in nwjs-client.js

- ```cd draw```
- On mac: ```nwjs .```
- On linux: ```nw .```

Note: On mac, nwjs is my alias for the nwjs application. For example, on a mac the full path is probably ```/Applications/nwjs.app/Contents/MacOS/nwjs```.

# Transitioning from Nan v1 to Nan v2

There are significant API changes from nan v1 to nan v2. Currently
node.js requires nan v2. To update nan and other utils to latest version of node and nan:

- ```npm install upgrade-utils -g```
- ```upgrade-utils```
- see upgrade-log.html for description of what will be updated
  automatically
- ```upgrade-utils --update```
- ```npm install nan@latest --save```
- rebuild ```make/make-nwjs```
