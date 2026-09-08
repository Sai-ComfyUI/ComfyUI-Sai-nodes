import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const COMMAND_ID = "sai.restartComfyUI";
const RESTART_ROUTE = "/sai/restart";

function toast(severity, summary, detail, life = 4000) {
    app.extensionManager.toast.add({ severity, summary, detail, life });
}

function sleep(milliseconds) {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function waitForRestart() {
    let observedOffline = false;

    for (let attempt = 0; attempt < 90; attempt += 1) {
        await sleep(1000);
        try {
            const response = await api.fetchApi("/system_stats", {
                cache: "no-store",
            });
            if (observedOffline && response.ok) {
                window.location.reload();
                return;
            }
        } catch {
            observedOffline = true;
        }
    }

    toast(
        "warn",
        "ComfyUI restart",
        "The service did not reconnect automatically. Refresh the page manually.",
        10000,
    );
}

async function restartComfyUI() {
    const confirmed = await app.extensionManager.dialog.confirm({
        title: "Restart ComfyUI?",
        message:
            "Running jobs will be interrupted. The page will reload after the local service is ready.",
    });
    if (!confirmed) {
        return;
    }

    try {
        const response = await api.fetchApi(RESTART_ROUTE, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}",
        });
        if (!response.ok) {
            throw new Error(await response.text());
        }

        toast(
            "info",
            "Restarting ComfyUI",
            "Waiting for the local service to come back…",
            6000,
        );
        void waitForRestart();
    } catch (error) {
        toast(
            "error",
            "ComfyUI restart failed",
            error instanceof Error ? error.message : String(error),
            10000,
        );
    }
}

app.registerExtension({
    name: "sai.serviceRestart",
    commands: [
        {
            id: COMMAND_ID,
            label: "Restart ComfyUI (Sai)",
            icon: "pi pi-refresh",
            function: restartComfyUI,
        },
    ],
    menuCommands: [
        {
            path: ["Sai"],
            commands: [COMMAND_ID],
        },
    ],
    actionBarButtons: [
        {
            icon: "pi pi-refresh",
            label: "Restart",
            tooltip: "Restart the local ComfyUI service",
            onClick: restartComfyUI,
        },
    ],
});
