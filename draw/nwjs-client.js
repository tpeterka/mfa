//
// nwjs standalone app client (acts as a node server)
//

// var addon = require('bindings')('draw');
var addon = require('./build/Release/draw');

addon.draw("../install/examples/nd/approx.out",
           nraw_pts,
           raw_pts,
           nctrl_pts,
           ctrl_pts,
           approx_pts,
           bbs);
draw();
