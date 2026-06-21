import QtQuick

// DeltaFlash — wrap a value Text/cell; when the bound `value` changes, the cell
// flashes SUCCESS (went up) or DANGER (went down) then fades back — for QC
// metrics / live stats that update during a run. Put your value Text inside.
//
//   DeltaFlash { value: Qc.linkRatio
//       MetricStat { label: "LINK RATIO"; value: Qc.linkRatioStr } }
Item {
    id: root
    property real value: 0
    property real _prev: value
    default property alias content: holder.data

    implicitWidth: holder.childrenRect.width
    implicitHeight: holder.childrenRect.height

    Rectangle {                       // flash wash behind the content
        id: wash
        anchors.fill: parent
        anchors.margins: -4
        radius: 4
        color: "transparent"
        opacity: 0
    }
    Item { id: holder; anchors.fill: parent }

    onValueChanged: {
        if (Theme.reducedMotion) { _prev = value; return }
        wash.color = value >= _prev ? Theme.palette.SUCCESS : Theme.palette.DANGER;
        _prev = value;
        flash.restart();
    }
    SequentialAnimation {
        id: flash
        NumberAnimation { target: wash; property: "opacity"; to: 0.16; duration: 110; easing.type: Easing.OutCubic }
        NumberAnimation { target: wash; property: "opacity"; to: 0.0;  duration: 420; easing.type: Easing.InCubic }
    }
}
