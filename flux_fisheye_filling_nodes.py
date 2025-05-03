import torch
import numpy as np

# reprojection helpers
from .reprojection_nodes import Projection, ReprojectImage, TransformToMatrix

# Try importing FluxInpainting and capture any ImportError
import sys, os, logging
here = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if here not in sys.path:
    sys.path.append(here)

try:
    from custom_nodes.inpainting_flux.nodes import FluxNF4Inpainting as FluxInpainting
    _flux_import_error = None
except ImportError as e:
    FluxInpainting = None
    _flux_import_error = e
    logging.error(f"[OutpaintAnyProjection] could not import FluxNF4Inpainting: {e}")

class OutpaintAnyProjection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":                ("IMAGE",),
                # — base canvas reprojection —
                "input_projection":     (Projection.PROJECTIONS, {"tooltip": "source projection"}),
                "input_horiz_fov":      ("FLOAT",  {"default": 90.0,"min": 0.0,  "max": 360.0}),
                "output_projection":    (Projection.PROJECTIONS, {"tooltip": "final projection"}),
                "output_horiz_fov":     ("FLOAT",  {"default":180.0,"min": 0.0,  "max": 360.0}),
                "output_width":         ("INT",    {"default":4096, "min":1,    "max":16384}),
                "output_height":        ("INT",    {"default":4096, "min":1,    "max":16384}),
                # — patch extraction / outpaint —
                "patch_projection":     (Projection.PROJECTIONS, {"tooltip": "patch projection"}),
                "patch_horiz_fov":      ("FLOAT",  {"default": 90.0,"min":1.0,  "max":180.0}),
                "patch_res":            ("INT",    {"default":1024, "min":1,    "max":4096}),
                "patch_phi":            ("FLOAT",  {"default":45.0, "min":-180.0,"max":180.0}),
                "patch_theta":          ("FLOAT",  {"default": 0.0, "min": -90.0,"max": 90.0}),
                # — Flux NF4 inpainting settings —
                "prompt":               ("STRING", {"multiline": True}),
                "num_inference_steps":  ("INT",    {"default":50,   "min":10,   "max":60}),
                "cached":               ("BOOLEAN",{"default":False}),
                "guidance_scale":       ("FLOAT",  {"default":30.0, "min":0.1,  "max":30.0}),
                # — optional feathering on patch edges —
                "mask_blur":            ("INT",    {"default":30,   "min":0,    "max":512}),
            },
            "optional": {
                "mask":                ("MASK",),
                "debug":               ("BOOLEAN", {"default": False, "tooltip": "If true, skip inpainting and output patch"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("final_image", "needs_inpaint_mask")
    FUNCTION      = "outpaint_any"
    CATEGORY      = "Camera/Outpainting"

    def outpaint_any(
        self,
        image,
        input_projection, input_horiz_fov,
        output_projection, output_horiz_fov, output_width, output_height,
        patch_projection, patch_horiz_fov, patch_res, patch_phi, patch_theta,
        prompt, num_inference_steps, cached, guidance_scale,
        mask_blur,
        mask=None,
        debug=False
    ):
        # Immediately fail if Flux isn't available
        if _flux_import_error is not None:
            raise RuntimeError(
                f"FluxNF4Inpainting is not available: {_flux_import_error}\n"
                "Please install or fix your inpainting_flux package."
            )

        def normalize_mask(m: torch.Tensor):
            # squeeze away any extra leading dims until shape is [B, H, W]
            while m.dim() > 3:
                m = m.squeeze(1)
            return m


        reproj = ReprojectImage()

        # 1) Base canvas reprojection
        base_img, base_mask = reproj.reproject_image(
            image,
            input_horiz_fov, output_horiz_fov,
            input_projection, output_projection,
            output_width, output_height,
            feathering=0,
            mask=mask
        )
        base_mask = normalize_mask(base_mask)
        # base_img: (1, H, W, C), base_mask: (1, H, W)

        # 2) Rotation for patch
        t2m   = TransformToMatrix()
        rot_m = t2m.generate_matrix(0, 0, 0, patch_theta, patch_phi)

        # 3) Extract patch
        patch_img, patch_mask = reproj.reproject_image(
            base_img,
            output_horiz_fov, patch_horiz_fov,
            output_projection, patch_projection,
            patch_res, patch_res,
            feathering=mask_blur,
            mask=base_mask,
            transform_matrix=rot_m
        )
        patch_mask = normalize_mask(patch_mask)
        # patch_img: (1, P, P, C), patch_mask: (1, P, P)

        # DEBUG: return patch and mask directly

        # 4) Flux NF4 Inpainting
        if not debug:
            flux = FluxInpainting()
            inpainted_patch, = flux.inpainting(
                prompt,
                patch_img,
                patch_mask,
                num_inference_steps,
                cached,
                guidance_scale
            )
        inpainted_patch=torch.ones_like(patch_img) * 0.5 if debug else inpainted_patch
        patch_mask = torch.ones_like(patch_mask) * 1 if debug else patch_mask
        # 5) Reproject inpainted patch back
       
        back_img, back_mask = reproj.reproject_image(
        inpainted_patch,
        patch_horiz_fov, output_horiz_fov,
        patch_projection, output_projection,
        output_width, output_height,
        inverse=True,
        transform_matrix=rot_m,
        feathering=0,
        )
        back_mask = normalize_mask(back_mask).bool()    # True where patch contributes

        # original coverage: True = had data, False = hole
        orig_covered = ~base_mask.bool()
        base_img=base_img * orig_covered.unsqueeze(-1)
        # fill only holes where back_mask is False
        filled = back_img * (~back_mask.unsqueeze(-1))*base_mask.unsqueeze(-1)
        final_img = base_img+filled

        # anything that’s still a hole after back‐projection needs inpaint
        needs_inpaint = (~((orig_covered) | (~back_mask))).to(torch.float32)

        return final_img, needs_inpaint

# register
NODE_CLASS_MAPPINGS = {
    "OutpaintAnyProjection": OutpaintAnyProjection
}
