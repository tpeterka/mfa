// geometry
var sites          = [];
var verts          = [];
var num_face_verts = [];
var vols           = [];
var tet_verts      = [];
var bbs            = [];

// run the addon
var addon = require('bindings')('draw');

addon.draw("./del.out",
           sites,
           verts,
           num_face_verts,
           vols,
           tet_verts,
           bbs);

// set up a websocket server
var WebSocketServer = require('ws').Server;
wss = new WebSocketServer({
    port      : 8080,
    binaryType: 'arraybuffer',
    perMessageDeflate: false                 // needed to fix a bug in ws for sending binary data
                                             // see https://github.com/websockets/ws/issues/523
});
wss.on('connection', function(ws) {
    ws.on('message', function(message) {
        console.log('received: %s', message);
    });

    // set ArrayBuffer through a DataView for each variable

    // sites
    var buffer = new ArrayBuffer(sites.length * 4);
    var view = new DataView(buffer);
    for (i = 0; i < sites.length; i++)
        view.setFloat32(i * 4, sites[i]);
    ws.send(buffer, {binary: true, mask: false});

    // verts
    var buffer = new ArrayBuffer(verts.length * 4);
    var view = new DataView(buffer);
    for (i = 0; i < verts.length; i++)
        view.setFloat32(i * 4, verts[i]);
    ws.send(buffer, {binary: true, mask: false});

    // num_face_verts
    var buffer = new ArrayBuffer(num_face_verts.length * 4);
    var view = new DataView(buffer);
    for (i = 0; i < num_face_verts.length; i++)
        view.setInt32(i * 4, num_face_verts[i]);
    ws.send(buffer, {binary: true, mask: false});

    // vols
    var buffer = new ArrayBuffer(vols.length * 4);
    var view = new DataView(buffer);
    for (i = 0; i < vols.length; i++)
        view.setFloat32(i * 4, vols[i]);
    ws.send(buffer, {binary: true, mask: false});

    // tet_verts
    var buffer = new ArrayBuffer(tet_verts.length * 4);
    var view = new DataView(buffer);
    for (i = 0; i < tet_verts.length; i++)
        view.setFloat32(i * 4, tet_verts[i]);
    ws.send(buffer, {binary: true, mask: false});

    // bbs
    var buffer = new ArrayBuffer(bbs.length * 4);
    var view = new DataView(buffer);
    for (i = 0; i < bbs.length; i++)
        view.setFloat32(i * 4, bbs[i]);
    ws.send(buffer, {binary: true, mask: false});

});
console.log('WS server running at http://127.0.0.1:8080/');
