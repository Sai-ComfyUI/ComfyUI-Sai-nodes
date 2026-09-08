import { app } from "../../scripts/app.js";

const NODE_ID = "LabeledImageCollage_Sai";
const STORAGE_KEY = "sai.labeledImageCollage.presets.v1";
const NO_PRESET = "(no preset)";
const PRESET_FIELDS = [
    "labels",
    "layout",
    "items_per_line",
    "size_mode",
    "cell_width",
    "cell_height",
    "cross_axis_size",
    "spacing",
    "outer_margin",
    "background_color",
    "label_placement",
    "label_position",
    "text_direction",
    "font_size",
    "font_color",
    "label_background_color",
    "label_background_extent",
    "label_padding",
    "label_margin",
    "outline_color",
    "outline_width",
    "font_name",
    "font_path",
];

function toast(severity, summary, detail) {
    app.extensionManager.toast.add({ severity, summary, detail, life: 4000 });
}

function loadPresets() {
    try {
        const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}");
        return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
    } catch {
        return {};
    }
}

function savePresets(presets) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
}

function presetNames() {
    return Object.keys(loadPresets()).sort((left, right) => left.localeCompare(right));
}

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function refreshPresetWidget(widget, selected = NO_PRESET) {
    widget.options.values = [NO_PRESET, ...presetNames()];
    widget.value = widget.options.values.includes(selected) ? selected : NO_PRESET;
}

function capturePreset(node) {
    return Object.fromEntries(
        PRESET_FIELDS.flatMap((name) => {
            const widget = findWidget(node, name);
            return widget ? [[name, widget.value]] : [];
        }),
    );
}

function applyPreset(node, preset) {
    for (const [name, value] of Object.entries(preset)) {
        if (!PRESET_FIELDS.includes(name)) {
            continue;
        }
        const widget = findWidget(node, name);
        if (!widget) {
            continue;
        }
        widget.value = value;
        widget.callback?.(value);
    }
    node.setDirtyCanvas(true, true);
}

function attachPresetControls(node) {
    if (node.__saiCollagePresetsAttached) {
        return;
    }
    node.__saiCollagePresetsAttached = true;

    const presetWidget = node.addWidget(
        "combo",
        "preset",
        NO_PRESET,
        (name) => {
            if (name === NO_PRESET) {
                return;
            }
            const preset = loadPresets()[name];
            if (preset) {
                applyPreset(node, preset);
                toast("success", "Collage preset loaded", name);
            }
        },
        { values: [NO_PRESET, ...presetNames()], serialize: false },
    );

    node.addWidget(
        "button",
        "save preset",
        null,
        async () => {
            const suggested = presetWidget.value === NO_PRESET ? "" : presetWidget.value;
            const response = await app.extensionManager.dialog.prompt({
                title: "Save collage preset",
                message: "Preset name",
                defaultValue: suggested,
            });
            const name = typeof response === "string" ? response.trim() : "";
            if (!name || name === NO_PRESET) {
                return;
            }
            try {
                const presets = loadPresets();
                presets[name] = capturePreset(node);
                savePresets(presets);
                refreshPresetWidget(presetWidget, name);
                toast("success", "Collage preset saved", name);
            } catch (error) {
                toast("error", "Could not save collage preset", String(error));
            }
        },
        { serialize: false },
    );

    node.addWidget(
        "button",
        "delete preset",
        null,
        async () => {
            const name = presetWidget.value;
            if (name === NO_PRESET) {
                toast("info", "Collage preset", "Select a preset to delete.");
                return;
            }
            const confirmed = await app.extensionManager.dialog.confirm({
                title: "Delete collage preset?",
                message: name,
            });
            if (!confirmed) {
                return;
            }
            try {
                const presets = loadPresets();
                delete presets[name];
                savePresets(presets);
                refreshPresetWidget(presetWidget);
                toast("success", "Collage preset deleted", name);
            } catch (error) {
                toast("error", "Could not delete collage preset", String(error));
            }
        },
        { serialize: false },
    );

    node.setSize(node.computeSize());
}

app.registerExtension({
    name: "sai.labeledImageCollagePresets",

    nodeCreated(node) {
        if (node.constructor.comfyClass === NODE_ID || node.comfyClass === NODE_ID) {
            attachPresetControls(node);
        }
    },
});
