var text = {
    Open      : function() {
        var evt = document.createEvent("MouseEvents");
        evt.initEvent("click", true, false);
        theFile.dispatchEvent(evt);
    },
    RawData     : true,
    Visible     : true,
    Z0          : false,
    ApproxData  : true,
    ErrorData   : true,
    Boundaries  : true,
    WhitePaper  : false,
    ResetView   : false
};

window.onload = function() {
    var gui = new dat.GUI();

    var f1 = gui.addFolder('File');

    f1.add(text, 'Open').name('Open file');

    var f2 = gui.addFolder('Geometry');

    f2.add(text, 'RawData').onChange(function(value) {
        var object = scene.getObjectByName('raw_curves');
        object.visible = value;
    });

    var f4 = f2.addFolder('ControlData');

    f4.add(text, 'Visible').onChange(function(value) {
        var object = scene.getObjectByName('ctrl_pts');
        object.visible = value;
        var object = scene.getObjectByName('ctrl_curves');
        object.visible = value;
    });

    f4.add(text, 'Z0').onChange(function(value) {
        var object = scene.getObjectByName('ctrl_pts');
        scene.remove(object);
        var object = scene.getObjectByName('ctrl_curves');
        scene.remove(object);
        ctrl_pts_z0 = value;
        create_ctrl_geometry();
    });

    f2.add(text, 'ApproxData').onChange(function(value) {
        var object = scene.getObjectByName('approx_curves');
        object.visible = value;
    });

    f2.add(text, 'ErrorData').onChange(function(value) {
        var object = scene.getObjectByName('error_curves');
        object.visible = value;
    });

    f2.add(text, 'Boundaries').onChange(function(value) {
        var object = scene.getObjectByName('bbs');
        object.visible = value;
    });

    var f3 = gui.addFolder('Rendering');

    f3.add(text, 'WhitePaper').onChange(function(value) {
        white_paper = value;
        if (white_paper)
        {
            renderer.setClearColor('white', 1);

            var object = scene.getObjectByName('raw_curves');
            object.material.color.set('blue');
            object.material.linewidth = 3;
            var object = scene.getObjectByName('ctrl_pts');
            object.material.color.set('gray');
            object.material.size *= 2;
            object.material.linewidth = 3;
            var object = scene.getObjectByName('ctrl_curves');
            object.material.color.set('gray');
            var object = scene.getObjectByName('approx_curves');
            object.material.color.set('red');
            object.material.linewidth = 3;
        }
        else
        {
            renderer.setClearColor('black', 1);

            var object = scene.getObjectByName('raw_curves');
            object.material.color.set('white');
            var object = scene.getObjectByName('ctrl_pts');
            object.material.color.set('yellow');
            var object = scene.getObjectByName('ctrl_curves');
            object.material.color.set('yellow');
            var object = scene.getObjectByName('approx_curves');
            object.material.color.set('cyan');
        }

    });

    f3.add(text, 'ResetView').onChange(function(value) {
        controls.reset();
        controls.target.set(center[0], center[1], center[2]);
        text.ResetView = false;
    });
};
