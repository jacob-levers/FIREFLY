import QtQuick

// Decorative "single-molecule" field for the landing left rail: drifting glowing
// particles (the emitters), slow inferno nebula blooms (a density / max-projection
// map) and the occasional cyan detection crosshair (the localiser finding a spot).
// Additive ('lighter') compositing over the near-black rail.  Pauses entirely
// under reduce-motion or when `active` is false (off the landing).
Item {
    id: root
    property bool active: true
    readonly property bool running: active && !Theme.reducedMotion && visible
                                    && width > 0 && height > 0
    clip: true

    // ── state (plain JS arrays; mutated in place, repainted manually) ────────
    property var _parts: []
    property var _blobs: []
    property var _marks: []
    property real _t: 0
    property real _lastSpawn: 0
    // emitter colours: luminous blue, detection cyan, near-white
    readonly property var _pal: [[88, 166, 255], [39, 192, 232], [230, 237, 243]]
    readonly property var _palStr: ["rgb(88,166,255)", "rgb(39,192,232)", "rgb(230,237,243)"]
    // nebula colours from the inferno ramp
    readonly property var _bp: [[147, 38, 103], [221, 81, 58], [246, 166, 35], [66, 10, 104]]

    function _rand() { return Math.random(); }

    function _init() {
        var w = width, h = height;
        if (w <= 0 || h <= 0) { _parts = []; _blobs = []; _marks = []; return; }
        var P = [];
        for (var i = 0; i < 52; i++) {
            var ci = (_rand() * _pal.length) | 0;
            P.push({ x: _rand() * w, y: _rand() * h,
                     r: 0.6 + _rand() * 2.1,
                     vx: _rand() - 0.5, vy: _rand() - 0.5,
                     ph: _rand() * 6.28, tw: 0.5 + _rand() * 1.4,
                     ci: ci, c: _pal[ci] });
        }
        _parts = P;
        var B = [];
        for (var j = 0; j < 4; j++)
            B.push({ x: _rand() * w, y: _rand() * h,
                     R: 110 + _rand() * 120,
                     vx: (_rand() - 0.5) * 0.2, vy: (_rand() - 0.5) * 0.2,
                     c: _bp[j % _bp.length], ph: _rand() * 6.28 });
        _blobs = B;
        _marks = [];
        _t = 0; _lastSpawn = 0;
    }

    function _step() {
        var w = width, h = height, dt = 2.0;     // ~32ms tick ≈ 2× the 16ms design frame
        _t += 0.032;
        var i, o;
        for (i = 0; i < _blobs.length; i++) {
            o = _blobs[i];
            o.x += o.vx * dt; o.y += o.vy * dt;
            if (o.x < -o.R) o.x = w + o.R; else if (o.x > w + o.R) o.x = -o.R;
            if (o.y < -o.R) o.y = h + o.R; else if (o.y > h + o.R) o.y = -o.R;
        }
        for (i = 0; i < _parts.length; i++) {
            o = _parts[i];
            o.x += o.vx * 0.25 * dt; o.y += o.vy * 0.25 * dt;
            if (o.x < -12) o.x = w + 12; else if (o.x > w + 12) o.x = -12;
            if (o.y < -12) o.y = h + 12; else if (o.y > h + 12) o.y = -12;
        }
        if (_t - _lastSpawn > 0.55 && _parts.length) {
            _lastSpawn = _t;
            _marks.push({ p: _parts[(_rand() * _parts.length) | 0], t0: _t });
            if (_marks.length > 6) _marks.shift();
        }
    }

    onWidthChanged:  { _init(); cv.requestPaint(); }
    onHeightChanged: { _init(); cv.requestPaint(); }
    Component.onCompleted: { _init(); cv.requestPaint(); }

    Timer {
        interval: 32; repeat: true; running: root.running
        onTriggered: { root._step(); cv.requestPaint(); }
    }

    Canvas {
        id: cv
        anchors.fill: parent
        // GPU-backed canvas for the continuously-animated field (falls back to the
        // image target automatically under the software backend / tests)
        renderTarget: Canvas.FramebufferObject
        onPaint: {
            var ctx = getContext("2d");
            ctx.clearRect(0, 0, width, height);
            ctx.globalCompositeOperation = "lighter";
            var t = root._t, i, o, g, x, y;
            // nebula blooms
            for (i = 0; i < root._blobs.length; i++) {
                o = root._blobs[i];
                var pulse = 0.06 + 0.035 * (0.5 + 0.5 * Math.sin(t * 0.5 + o.ph));
                g = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, o.R);
                g.addColorStop(0, "rgba(" + o.c[0] + "," + o.c[1] + "," + o.c[2] + "," + pulse + ")");
                g.addColorStop(1, "rgba(" + o.c[0] + "," + o.c[1] + "," + o.c[2] + ",0)");
                ctx.fillStyle = g;
                ctx.beginPath(); ctx.arc(o.x, o.y, o.R, 0, 6.2832); ctx.fill();
            }
            // emitters (soft glow + crisp core, twinkling).  Build ONE unit
            // glow gradient per colour for the whole frame and reuse it for every
            // particle via the canvas transform (the gradient is mapped by the CTM
            // at fill time) — was 52 createRadialGradient() calls per frame.
            var ug = [];
            for (i = 0; i < root._pal.length; i++) {
                var pc = root._pal[i];
                var gg = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
                gg.addColorStop(0, "rgba(" + pc[0] + "," + pc[1] + "," + pc[2] + ",0.45)");
                gg.addColorStop(1, "rgba(" + pc[0] + "," + pc[1] + "," + pc[2] + ",0)");
                ug.push(gg);
            }
            for (i = 0; i < root._parts.length; i++) {
                o = root._parts[i];
                var a = 0.4 + 0.6 * (0.5 + 0.5 * Math.sin(t * o.tw + o.ph));
                var R = o.r * 6;
                ctx.globalAlpha = a;                 // twinkle (× the 0.45 glow base)
                ctx.save();
                ctx.translate(o.x, o.y); ctx.scale(R, R);
                ctx.fillStyle = ug[o.ci];
                ctx.beginPath(); ctx.arc(0, 0, 1, 0, 6.2832); ctx.fill();
                ctx.restore();
                ctx.fillStyle = root._palStr[o.ci]; // crisp core
                ctx.beginPath(); ctx.arc(o.x, o.y, o.r, 0, 6.2832); ctx.fill();
            }
            ctx.globalAlpha = 1;
            // detection crosshairs (corner brackets, fade in/out)
            for (i = 0; i < root._marks.length; i++) {
                o = root._marks[i];
                var age = (t - o.t0) / 1.4;
                if (age > 1) continue;
                var al = Math.sin(age * Math.PI) * 0.7;
                var s = 10 + age * 4, c = 4; x = o.p.x; y = o.p.y;
                ctx.strokeStyle = "rgba(39,192,232," + al + ")";
                ctx.lineWidth = 1;
                var corners = [[-1, -1], [1, -1], [-1, 1], [1, 1]];
                for (var k = 0; k < 4; k++) {
                    var sx = corners[k][0], sy = corners[k][1];
                    ctx.beginPath();
                    ctx.moveTo(x + sx * s, y + sy * s - sy * c);
                    ctx.lineTo(x + sx * s, y + sy * s);
                    ctx.lineTo(x + sx * s - sx * c, y + sy * s);
                    ctx.stroke();
                }
            }
            ctx.globalCompositeOperation = "source-over";
        }
    }
}
