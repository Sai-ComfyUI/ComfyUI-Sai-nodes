"""ComfyUI-Sai-nodes V3 extension entrypoint."""

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes.conditioning.flux2_klein_multi_reference import (
    Flux2KleinMultiReferenceLatent,
)
from .nodes.image.mytimemachine import (
    LoadMyTimeMachineModelSai,
    MyTimeMachineAgeTransformSai,
)
from .nodes.image.mytimemachine_training import (
    MyTimeMachineTrainingStatusSai,
    StartMyTimeMachineTrainingSai,
)
from .nodes.image.face_reframe import (
    MyTimeMachineFaceAlignSai,
    MyTimeMachineFaceRestoreSai,
)
from .nodes.image.labeled_collage import LabeledImageCollage
from .nodes.latent.lua_flux import LoadLuaFluxModel, LuaFluxLatentUpscale
from .server.restart import register_restart_route


WEB_DIRECTORY = "./web"


class SaiNodesExtension(ComfyExtension):
    """Register the public nodes provided by this package."""

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Flux2KleinMultiReferenceLatent,
            LoadMyTimeMachineModelSai,
            StartMyTimeMachineTrainingSai,
            MyTimeMachineTrainingStatusSai,
            MyTimeMachineFaceAlignSai,
            MyTimeMachineAgeTransformSai,
            MyTimeMachineFaceRestoreSai,
            LabeledImageCollage,
            LoadLuaFluxModel,
            LuaFluxLatentUpscale,
        ]

    @override
    async def on_load(self) -> None:
        register_restart_route()


async def comfy_entrypoint() -> SaiNodesExtension:
    return SaiNodesExtension()
