//
// websockets client
// connects to server and receives geometry data
//

log = document.getElementById("log");

// enum for message type
// sites         : 0
// verts         : 1
// num_face_verts: 2
// vols          : 3
// tet_verts     : 4
// bbs           : 5
var message_type = 0;
var num_message_types = 6;

if (window.WebSocket === undefined)
{
    console.log('Websockets not supported');
}
else
{
    // TODO: ...
    window.addEventListener("load", onLoad, false);
}

function onLoad()
{
    var wsUri = "ws://127.0.0.1:8080";

    websocket            = new WebSocket(wsUri);
    websocket.binaryType = 'arraybuffer';
    websocket.onopen     = function(evt) { onOpen(evt) };
    websocket.onclose    = function(evt) { onClose(evt) };
    websocket.onmessage  = function(evt) { onMessage(evt) };
    websocket.onerror    = function(evt) { onError(evt) };
}

function onOpen(evt)
{
    console.log('Connected to server');
}

function onClose(evt)
{
    console.log('Connection closed');
}

function onMessage(evt)
{
    var view = new DataView(evt.data);

    if (message_type == 0)
    {
        for (i = 0; i < view.byteLength / 4; i++)
            sites.push(view.getFloat32(i * 4));
    }
    if (message_type == 1)
    {
        for (i = 0; i < view.byteLength / 4; i++)
            verts.push(view.getFloat32(i * 4));
    }
    if (message_type == 2)
    {
        for (i = 0; i < view.byteLength / 4; i++)
            num_face_verts.push(view.getInt32(i * 4));
    }
    if (message_type == 3)
    {
        for (i = 0; i < view.byteLength / 4; i++)
            vols.push(view.getFloat32(i * 4));
    }
    if (message_type == 4)
    {
        for (i = 0; i < view.byteLength / 4; i++)
            tet_verts.push(view.getFloat32(i * 4));
    }
    if (message_type == 5)
    {
        for (i = 0; i < view.byteLength / 4; i++)
            bbs.push(view.getFloat32(i * 4));
    }

    if (message_type == 5)
        draw();

    message_type++;

    // debug
//     console.log('message_type:', message_type);
//     console.log('received sites:', sites);
//     console.log('received verts:', verts);
//     console.log('received num_face_verts:', num_face_verts);
//     console.log('received vols:', vols);
//     console.log('received tet_verts:', tet_verts);
//     console.log('received bbs:', bbs);
}

function onError(evt)
{
    console.log('Communication error');
}

