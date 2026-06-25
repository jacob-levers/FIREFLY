import QtQuick
import QtQuick.Layouts

// Landing-page action tile: accent icon-chip + title + description, hover-lift +
// accent border, a hover-revealed arrow affordance. Matches the after-landing
// mockup. Emits clicked().
Rectangle {
    id: root
    property string icon: ""
    property string title: ""
    property string desc: ""
    signal clicked()

    readonly property var pal: Theme.palette
    readonly property var sc: Theme.scale
    readonly property int dur: Theme.reducedMotion ? 0 : 140

    implicitHeight: 96
    radius: sc.radius2xl
    color: hov.hovered ? pal.PANEL_ALT : pal.PANEL
    border.width: 1
    border.color: hov.hovered ? pal.ACC : pal.BORDER

    Behavior on color { ColorAnimation { duration: root.dur } }
    Behavior on border.color { ColorAnimation { duration: root.dur } }

    // hover-lift (a transform, so it doesn't fight the layout position)
    transform: Translate {
        id: lift
        y: hov.hovered ? -3 : 0
        Behavior on y { NumberAnimation { duration: root.dur; easing.type: Easing.OutCubic } }
    }

    RowLayout {
        anchors.fill: parent
        anchors.margins: sc.sp6
        spacing: sc.sp6

        Rectangle {                       // icon chip
            Layout.alignment: Qt.AlignVCenter
            width: 38; height: 38
            radius: sc.radius2xl
            color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, hov.hovered ? 0.16 : 0.10)
            border.width: 1
            border.color: Qt.rgba(pal.ACC.r, pal.ACC.g, pal.ACC.b, 0.22)
            Behavior on color { ColorAnimation { duration: root.dur } }
            Icon { anchors.centerIn: parent; name: root.icon; color: pal.ACC; size: 20 }
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignVCenter
            spacing: sc.sp1
            Text {
                text: root.title; color: pal.TXT
                font.pixelSize: sc.textLg; font.bold: true
            }
            Text {
                text: root.desc; color: pal.TXT_MUTED
                font.pixelSize: sc.textSm
                Layout.fillWidth: true; wrapMode: Text.WordWrap
            }
        }

        Icon {                            // hover affordance
            Layout.alignment: Qt.AlignTop
            name: "arrow-up-right"; color: pal.TXT_MUTED; size: 16
            opacity: hov.hovered ? 1 : 0
            Behavior on opacity { NumberAnimation { duration: root.dur } }
        }
    }

    HoverHandler { id: hov }
    TapHandler { onTapped: root.clicked() }
}
