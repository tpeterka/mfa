# Diy2-draw

Diy2-draw demonstrates how to develop a webGL renderer to draw geometry contained in a diy2 output file. This particular example draws the ouput of [tess2](https://github.com/diatomic/tess2), a Delaunay and Voronoi tessellation library based on [diy2](https://github.com/diatomic/diy2).

The key features are:

- C++, DIY, MPI server using a node.js addon reads the data
- two versions of client-server connection
    - separate client and server connected by websockets
    - combined client and server in a standalone app using node webkit (nwjs)
- Javascript / webGL rendering on the client (using three.js rendering library instead of webGL directly)

The steps to build and run the example are below.

# Install dependencies:

- [node.js](https://nodejs.org/)
- [nwjs](https://github.com/nwjs/nw.js/)
- [three.js](http://threejs.org/) (for 3d rendering, a library wrapping webGL) Version 71 is bundled in the ```3rdparty``` directory.
- [dat.gui](https://code.google.com/p/dat-gui/) (for GUI controls) Version 0.5 is bundled in the ```3rdparty``` directory.
- [ws](https://github.com/websockets/ws) (websocket for node.js, needed for client-server communication, not needed for standalone nwjs)
    - ```cd src; npm install ws``` (npm is included with node.js)
- nan
    - ```cd src; npm install nan```
- bindings
    - ```cd src; npm install bindings```
- node-gyp
    - ```cd src; npm install -g node-gyp```
- nw-gyp
    - ```cd src; npm install -g nw-gyp```
- diy2 and tess2 (for this particular tessellation example)
    - [diy2](https://github.com/diatomic/diy2)
    - [tess2](https://github.com/diatomic/tess2)

# To create a node.js addon:

See [Node.js addon instructions](https://github.com/nodejs/node-addon-examples)

- Add or edit existing package.json file
- Add or edit existing binding.gyp file
- Add or edit a .cc or .cxx file for the node addon C/C++ server code
- Add or edit one or more.js files for the javascript client code
- Add or edit index.html file

# To build the executable:

One can either build a node.js server and a client to run in a browser, or a single standalone nwjs application. The former requires running a server and connecting to it from a client in a browser, while the latter is just one command to launch an app that looks and feels like a browser. The client-server node method can be used to distribute to others so that all the dependencies above are not needed locally. The nwjs standalone method is good for development and testing, but requires locally installing all the dependencies.

```cd src```

Edit the path names in binding.gyp to your own. Then,

- (client-server web application) To build a node program
    - ```CC=mpicc CXX=mpicxx node-gyp rebuild```
- (standalone application) To build an nwjs (node webkit) program (requires different compile command (nw-gyp) and the current version of nwjs needs to be specified because it cannot be found automatically
    - ```CC=mpicc CXX=mpicxx nw-gyp rebuild --target=0.12.2```
- The above commands are in ```make/make-node``` and ```make/make-nwjs```.

# To run the program:

Generate a test input file named ```del.out``` by running the script in tess2/examples/tess/TESS_TEST. Move del.out to the src directory of diy2-draw.

```cd src```

- (client-server web application)
    - in index.html, comment out nwjs-client.js and comment in node-client.js
    - server: ```node server.js```
    - client: In a browser, open index.html
- (standalone application)
    - in index.html, comment out node-client.js and comment in nwjs-client.js
    - On mac: ```nwjs .```
    - On linux: ```nw .```

Note: On mac, nwjs is my alias for the nwjs application. For example, on a mac the full path is probably ```/Applications/nwjs.app/Contents/MacOS/nwjs```.
