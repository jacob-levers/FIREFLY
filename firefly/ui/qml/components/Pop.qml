import QtQuick

// Pop — a tiny scale-in for things that ANNOUNCE themselves: a significance
// star, a done check-mark, a new badge, a QC pass. Scales 0.6→1 (no overshoot —
// house rule) + fades, once, on completion or when `shown` flips true.
//
//   Pop { Icon { name: "circle-check"; color: pal.SUCCESS; size: 16 } }
//   Pop { shown: row.status === "done"; Icon { name: "check"; ... } }
Item {
    id: root
    property bool shown: false
    property int  duration: 160
    default property alias content: holder.data

    implicitWidth:  holder.childrenRect.width
    implicitHeight: holder.childrenRect.height

    Component.onCompleted: root.shown = true

    Item {
        id: holder
        width: root.width; height: root.height
        opacity: root.shown ? 1 : 0
        scale:   root.shown ? 1 : 0.6
        Behavior on opacity { NumberAnimation { duration: Theme.reducedMotion ? 0 : root.duration; easing.type: Easing.OutCubic } }
        Behavior on scale   { NumberAnimation { duration: Theme.reducedMotion ? 0 : root.duration; easing.type: Easing.OutCubic } }
    }
}
