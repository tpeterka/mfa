var text = {
    Open      : function() {
        var evt = document.createEvent("MouseEvents");
        evt.initEvent("click", true, false);
        theFile.dispatchEvent(evt);
    },
    RawData     : true,
    ControlData : true,
    ApproxData  : true,
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
        // var object = scene.getObjectByName('raw_pts');
        // object.visible = value;
        var object = scene.getObjectByName('raw_curve');
        object.visible = value;
    });

    f2.add(text, 'ControlData').onChange(function(value) {
        var object = scene.getObjectByName('ctrl_pts');
        object.visible = value;
        var object = scene.getObjectByName('ctrl_curve');
        object.visible = value;
    });

    f2.add(text, 'ApproxData').onChange(function(value) {
        var object = scene.getObjectByName('approx_curve');
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
            // var object = scene.getObjectByName('raw_pts');
            // object.material.color.set('black');
            var object = scene.getObjectByName('raw_curve');
            object.material.color.set('black');
            var object = scene.getObjectByName('ctrl_pts');
            object.material.color.set('red');
            var object = scene.getObjectByName('ctrl_curve');
            object.material.color.set('red');
            var object = scene.getObjectByName('approx_curve');
            object.material.color.set('green');
        }
        else
        {
            renderer.setClearColor('black', 1);
            // var object = scene.getObjectByName('raw_pts');
            // object.material.color.set('white');
            var object = scene.getObjectByName('raw_curve');
            object.material.color.set('white');
            var object = scene.getObjectByName('ctrl_pts');
            object.material.color.set('yellow');
            var object = scene.getObjectByName('ctrl_curve');
            object.material.color.set('yellow');
            var object = scene.getObjectByName('approx_curve');
            object.material.color.set('cyan');
        }

    });

    f3.add(text, 'ResetView').onChange(function(value) {
        controls.reset();
        controls.target.set(center[0], center[1], center[2]);
        text.ResetView = false;
    });
};
