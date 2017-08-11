PT_SIZE_FACTOR = .005;             // used to compute point size

var center = [], size, sph_rad;
var scene, camera, renderer, controls, stats, light;

function draw()
{
    // init rendering
    if (!Detector.webgl)
        Detector.addGetWebGLMessage();
    init_render();

    // create scene geometry
    create_geometry();

    // render
    render();
    animate();
}

//
// create bounding box geometry
//
function create_bb_geometry()
{
    // bounding boxes
    material = new THREE.LineBasicMaterial({ color: 'magenta' });
    boxes = new THREE.Object3D();
    boxes.name = 'bbs';
    for (i = 0; i < bbs.length / 6; i++)
    {

        geometry = new THREE.Geometry();
        geometry.vertices.push(
	    new THREE.Vector3(bbs[6 * i    ],
                              bbs[6 * i + 1],
                              bbs[6 * i + 2]),
	    new THREE.Vector3(bbs[6 * i + 3],
                              bbs[6 * i + 1],
                              bbs[6 * i + 2]),
	    new THREE.Vector3(bbs[6 * i + 3],
                              bbs[6 * i + 4],
                              bbs[6 * i + 2]),
	    new THREE.Vector3(bbs[6 * i    ],
                              bbs[6 * i + 4],
                              bbs[6 * i + 2]),
	    new THREE.Vector3(bbs[6 * i    ],
                              bbs[6 * i + 1],
                              bbs[6 * i + 2]));
        line = new THREE.Line(geometry, material);
        boxes.add(line);
        geometry = new THREE.Geometry();
        geometry.vertices.push(
	    new THREE.Vector3(bbs[6 * i    ],
                              bbs[6 * i + 1],
                              bbs[6 * i + 5]),
	    new THREE.Vector3(bbs[6 * i + 3],
                              bbs[6 * i + 1],
                              bbs[6 * i + 5]),
	    new THREE.Vector3(bbs[6 * i + 3],
                              bbs[6 * i + 4],
                              bbs[6 * i + 5]),
	    new THREE.Vector3(bbs[6 * i    ],
                              bbs[6 * i + 4],
                              bbs[6 * i + 5]),
	    new THREE.Vector3(bbs[6 * i    ],
                              bbs[6 * i + 1],
                              bbs[6 * i + 5]));
        line = new THREE.Line(geometry, material);
        boxes.add(line);
        geometry = new THREE.Geometry();
        geometry.vertices.push(
	    new THREE.Vector3(bbs[6 * i    ],
                              bbs[6 * i + 1],
                              bbs[6 * i + 2]),
	    new THREE.Vector3(bbs[6 * i    ],
                              bbs[6 * i + 1],
                              bbs[6 * i + 5]));
        line = new THREE.Line(geometry, material);
        boxes.add(line);
        geometry = new THREE.Geometry();
        geometry.vertices.push(
	    new THREE.Vector3(bbs[6 * i + 3],
                              bbs[6 * i + 1],
                              bbs[6 * i + 2]),
	    new THREE.Vector3(bbs[6 * i + 3],
                              bbs[6 * i + 1],
                              bbs[6 * i + 5]));
        line = new THREE.Line(geometry, material);
        boxes.add(line);
        geometry = new THREE.Geometry();
        geometry.vertices.push(
	    new THREE.Vector3(bbs[6 * i + 3],
                              bbs[6 * i + 4],
                              bbs[6 * i + 2]),
	    new THREE.Vector3(bbs[6 * i + 3],
                              bbs[6 * i + 4],
                              bbs[6 * i + 5]));
        line = new THREE.Line(geometry, material);
        boxes.add(line);
        geometry = new THREE.Geometry();
        geometry.vertices.push(
	    new THREE.Vector3(bbs[6 * i    ],
                              bbs[6 * i + 4],
                              bbs[6 * i + 2]),
	    new THREE.Vector3(bbs[6 * i    ],
                              bbs[6 * i + 4],
                              bbs[6 * i + 5]));
        line = new THREE.Line(geometry, material);
        boxes.add(line);
    }
    scene.add(boxes);
}

//
// create raw data geometry
//
function create_raw_geometry()
{
    raw = new THREE.Object3D();              // curves connecting raw data
    curve_material = new THREE.LineBasicMaterial({ color: 'white', linewidth: 2 });

    // 1-d path
    if (nraw_pts.length < 2)
        nraw_pts[1] = 1;

    // x-direction curves
    n = 0;                                   // index into points
    for (j = 0; j < nraw_pts[1]; j++)
    {
        // create points
        points = new THREE.Geometry();
        for (i = 0; i < nraw_pts[0]; i++)
        {
            point = new THREE.Vector3(raw_pts[3 * n],
                                      raw_pts[3 * n + 1],
                                      raw_pts[3 * n + 2]);
            points.vertices.push(point);
            n++;
        }

        // create the lines and add to the scene
        curve = new THREE.Line(points, curve_material)
        raw.add(curve)
    }

    // y-direction curves
    if (nraw_pts.length >= 2)
    {
        no = 0;                              // starting offset of curve point
        for (j = 0; j < nraw_pts[0]; j++)
        {
            // create points
            points = new THREE.Geometry();
            n = no;
            for (i = 0; i < nraw_pts[1]; i++)
            {
                point = new THREE.Vector3(raw_pts[3 * n    ],
                                          raw_pts[3 * n + 1],
                                          raw_pts[3 * n + 2]);
                points.vertices.push(point);
                n = n + nraw_pts[0];
            }
            no++;

            // create the lines and add to the scene
            curve = new THREE.Line(points, curve_material)
            raw.add(curve)
        }
    }
    raw.material = curve_material;
    raw.name = 'raw_curves';
    scene.add(raw);
}

//
// create control data geometry
//
function create_ctrl_geometry()
{
    // 1-d path
    if (nctrl_pts.length < 2)
        nctrl_pts[1] = 1;

    // point cloud
    var point_material = new THREE.PointCloudMaterial({
        color: 'yellow',
        size: sph_rad
    });
    points = new THREE.Geometry();

    n = 0;                                   // index into points
    for (j = 0; j < nctrl_pts[1]; j++)
    {
        // create points
        for (i = 0; i < nctrl_pts[0]; i++)
        {
            var point;
            if (ctrl_pts_z0)
                point = new THREE.Vector3(ctrl_pts[3 * n], ctrl_pts[3 * n + 1], 0.0);
            else
                point = new THREE.Vector3(ctrl_pts[3 * n], ctrl_pts[3 * n + 1], ctrl_pts[3 * n + 2]);

            points.vertices.push(point);
            n++;
        }
    }

    pointSet = new THREE.PointCloud(points, point_material);
    pointSet.name = 'ctrl_pts';
    scene.add(pointSet);

    // curves
    ctrl = new THREE.Object3D();             // curves connecting control points
    curve_material = new THREE.LineBasicMaterial({ color: 'yellow', linewidth: 2 });

    // x-direction curves
    n = 0;                                   // index into points
    for (j = 0; j < nctrl_pts[1]; j++)
    {
        // create points
        points = new THREE.Geometry();
        for (i = 0; i < nctrl_pts[0]; i++)
        {
            var point;
            if (ctrl_pts_z0)
                point = new THREE.Vector3(ctrl_pts[3 * n], ctrl_pts[3 * n + 1], 0.0);
            else
                point = new THREE.Vector3(ctrl_pts[3 * n], ctrl_pts[3 * n + 1], ctrl_pts[3 * n + 2]);

            points.vertices.push(point);
            n++;
        }

        // create the lines and add to the scene
        curve = new THREE.Line(points, curve_material)
        ctrl.add(curve);
    }

    // y-direction curves
    if (nctrl_pts.length >= 2)
    {
        no = 0;                              // starting offset of curve point
        for (j = 0; j < nctrl_pts[0]; j++)
        {
            // create points
            points = new THREE.Geometry();
            n = no;
            for (i = 0; i < nctrl_pts[1]; i++)
            {
                var point;
                if (ctrl_pts_z0)
                    point = new THREE.Vector3(ctrl_pts[3 * n], ctrl_pts[3 * n + 1], 0.0);
                else
                    point = new THREE.Vector3(ctrl_pts[3 * n], ctrl_pts[3 * n + 1], ctrl_pts[3 * n + 2]);

                points.vertices.push(point);
                n = n + nctrl_pts[0];
            }
            no++;

            // create the lines and add to the scene
            curve = new THREE.Line(points, curve_material)
            ctrl.add(curve);
        }
    }
    ctrl.material = curve_material;
    ctrl.name = 'ctrl_curves';
    scene.add(ctrl);
}

//
// create approximated geometry
//
function create_approx_geometry()
{
    approx = new THREE.Object3D();           // curves connecting approximated points
    curve_material = new THREE.LineBasicMaterial({ color: 'cyan', linewidth: 2 });

    // 1-d path
    if (nraw_pts.length < 2)
        nraw_pts[1] = 1;

    // x-direction curves
    n = 0;                                   // index into points
    for (j = 0; j < nraw_pts[1]; j++)
    {
        // create points
        points = new THREE.Geometry();
        for (i = 0; i < nraw_pts[0]; i++)
        {
            point = new THREE.Vector3(approx_pts[3 * n],
                                      approx_pts[3 * n + 1],
                                      approx_pts[3 * n + 2]);
            points.vertices.push(point);
            n++;
        }

        // create the lines and add to the scene
        curve = new THREE.Line(points, curve_material)
        approx.add(curve)
    }

    // y-direction curves
    if (nraw_pts.length >= 2)
    {
        no = 0;                              // starting offset of curve point
        for (j = 0; j < nraw_pts[0]; j++)
        {
            // create points
            points = new THREE.Geometry();
            n = no;
            for (i = 0; i < nraw_pts[1]; i++)
            {
                point = new THREE.Vector3(approx_pts[3 * n    ],
                                          approx_pts[3 * n + 1],
                                          approx_pts[3 * n + 2]);
                points.vertices.push(point);
                n = n + nraw_pts[0];
            }
            no++;

            // create the lines and add to the scene
            curve = new THREE.Line(points, curve_material)
            approx.add(curve);
        }
    }
    approx.material = curve_material;
    approx.name = 'approx_curves';
    scene.add(approx);
}

//
// create error geometry
//
function create_error_geometry()
{
    error = new THREE.Object3D();           // curves connecting error points
    curve_material = new THREE.LineBasicMaterial({ color: 'magenta', linewidth: 2 });

    // 1-d path
    if (nraw_pts.length < 2)
        nraw_pts[1] = 1;

    // x-direction curves
    n = 0;                                   // index into points
    for (j = 0; j < nraw_pts[1]; j++)
    {
        // create points
        points = new THREE.Geometry();
        for (i = 0; i < nraw_pts[0]; i++)
        {
            point = new THREE.Vector3(err_pts[3 * n],
                                      err_pts[3 * n + 1],
                                      err_pts[3 * n + 2]);
            points.vertices.push(point);
            n++;
        }

        // create the lines and add to the scene
        curve = new THREE.Line(points, curve_material)
        error.add(curve)
    }

    // y-direction curves
    if (nraw_pts.length >= 2)
    {
        no = 0;                              // starting offset of curve point
        for (j = 0; j < nraw_pts[0]; j++)
        {
            // create points
            points = new THREE.Geometry();
            n = no;
            for (i = 0; i < nraw_pts[1]; i++)
            {
                point = new THREE.Vector3(err_pts[3 * n    ],
                                          err_pts[3 * n + 1],
                                          err_pts[3 * n + 2]);
                points.vertices.push(point);
                n = n + nraw_pts[0];
            }
            no++;

            // create the lines and add to the scene
            curve = new THREE.Line(points, curve_material)
            error.add(curve);
        }
    }
    error.material = curve_material;
    error.name = 'error_curves';
    scene.add(error);
}

//
// create scene geometry
//
function create_geometry()
{
    create_raw_geometry();                   // input data
    create_ctrl_geometry();                  // output control data
    create_approx_geometry();                // approximated data (for rendering only)
    create_error_geometry();                 // error data (for rendering only)
    create_bb_geometry();                    // bounding boxes
}

//
// animate
//
function animate()
{
    requestAnimationFrame(animate);
    controls.update();
    render();
}

//
// render
//
function render()
{
    renderer.render(scene, camera);
    stats.update();
}

//
// initialize rendering
//
function init_render()
{
    var raw_min = [];
    var raw_max = [];

    // min, max of raw_pts
    for (i = 0; i < raw_pts.length / 3; i++)
    {
        if (i == 0)
        {
            raw_min[0] = raw_pts[3 * i    ];
            raw_min[1] = raw_pts[3 * i + 1];
            raw_min[2] = raw_pts[3 * i + 2];
            raw_max[0] = raw_pts[3 * i    ];
            raw_max[1] = raw_pts[3 * i + 1];
            raw_max[2] = raw_pts[3 * i + 2];
        }
        if (raw_pts[3 * i   ] < raw_min[0])
            raw_min[0] = raw_pts[3 * i   ];
        if (raw_pts[3 * i + 1] < raw_min[1])
            raw_min[1] = raw_pts[3 * i + 1];
        if (raw_pts[3 * i + 2] < raw_min[2])
            raw_min[2] = raw_pts[3 * i + 2];
        if (raw_pts[3 * i   ] > raw_max[0])
            raw_max[0] = raw_pts[3 * i   ];
        if (raw_pts[3 * i + 1] > raw_max[1])
            raw_max[1] = raw_pts[3 * i + 1];
        if (raw_pts[3 * i + 2] > raw_max[2])
            raw_max[2] = raw_pts[3 * i + 2];
    }

    // center of raw_pts
    center[0] = (raw_min[0] + raw_max[0]) / 2.0;
    center[1] = (raw_min[1] + raw_max[1]) / 2.0;
    center[2] = (raw_min[2] + raw_max[2]) / 2.0;

    // extent of raw_pts
    var sizes = [];
    sizes[0] = raw_max[0] - raw_min[0];
    sizes[1] = raw_max[1] - raw_min[1];
    sizes[2] = raw_max[2] - raw_min[2];
    size = sizes[0];                         // max size in any dimension
    if (sizes[1] > size)
        size = sizes[1];
    if (sizes[2] > size)
        size = sizes[2];

    var fov = 60;                            // "normal" field of view, neither wide-angle nor tele
    sph_rad = PT_SIZE_FACTOR * size;         // size of spheres to draw for raw_pts
    var fov_rad = fov * Math.PI / 180;       // fov in radians
    var dz = size / Math.tan(fov_rad / 2);   // z distance of camera from center of object

    // scene, camera, renderer
    scene = new THREE.Scene();

    // camera with light attached
    camera = new THREE.PerspectiveCamera(fov, window.innerWidth/window.innerHeight, 0.01, 10000);
    camera.position.set(center[0], center[1], center[2] + dz);
    light = new THREE.PointLight( 0xffffff, 1.0, 100 );
    camera.add(light);
    scene.add(camera);

    // renderer
    renderer = new THREE.WebGLRenderer();
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // controller
    controls = new THREE.TrackballControls(camera, renderer.domElement);
    controls.rotateSpeed = 1.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;
    controls.noZoom = false;
    controls.noPan = false;
    controls.staticMoving = true;
    controls.dynamicDampingFactor = 0.3;
    controls.keys = [ 65, 83, 68 ];
    controls.target.set(center[0], center[1], center[2]);
    controls.addEventListener('change', render);

    // stats
    stats = new Stats();
    stats.domElement.style.position = 'absolute';
    stats.domElement.style.top = '0px';
    stats.domElement.style.left = '0px';
    stats.domElement.style.zIndex = 1;
    document.body.appendChild(stats.domElement);

    window.addEventListener('resize', onWindowResize, false);

    render();
}

//
// window resize event handler
//
function onWindowResize()
{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    controls.handleResize();
    render();
}
