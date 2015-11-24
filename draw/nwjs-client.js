//
// nwjs standalone app client (acts as a node server)
//

var addon = require('bindings')('draw');

addon.draw("../install/examples/simple/approx.out",
           raw_pts,
           ctrl_pts,
           approx_pts,
           bbs);

draw();
