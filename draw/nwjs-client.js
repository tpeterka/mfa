//
// nwjs standalone app client (acts as a node server)
//

var addon = require('./build/Release/draw');

addon.draw(nw.App.argv[0],
           nraw_pts,
           raw_pts,
           nctrl_pts,
           ctrl_pts,
           approx_pts,
           err_pts,
           bbs);
draw();
