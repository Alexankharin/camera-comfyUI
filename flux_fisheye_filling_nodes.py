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
                "output_width":         ("INT",    {"default":4096, "min":1,    "max":8192}),
                "output_height":        ("INT",    {"default":4096, "min":1,    "max":8192}),
                # — patch extraction / outpaint —
                "patch_projection":     (Projection.PROJECTIONS, {"tooltip": "patch projection"}),
                "patch_horiz_fov":      ("FLOAT",  {"default": 90.0,"min":1.0,  "max":180.0}),
                "patch_res":            ("INT",    {"default":1024, "min":1,    "max":8192}),
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
       
        back_img, inpainted_patch_reproj_mask_raw = reproj.reproject_image( # Store raw mask output
        inpainted_patch,
        patch_horiz_fov, output_horiz_fov,
        patch_projection, output_projection,
        output_width, output_height,
        inverse=True,
        transform_matrix=rot_m,
        feathering=0,
        )
        # inpainted_patch_coverage_mask: 1.0 where the reprojected inpainted patch has content, 0.0 otherwise.
        inpainted_patch_coverage_mask = normalize_mask(back_mask) # Ensure it's float [0,1]

        # --- Define Masks based on Conventions ---
        # base_mask: 1.0 where original reprojected image has content, 0.0 for holes.
        # initial_hole_mask: 1.0 where original reprojected image has holes (inverse of base_mask).
        initial_hole_mask = 1.0 - base_mask
        # inpainted_patch_coverage_mask: 1.0 where reprojected inpainted patch has content.

        # --- Compositing Logic ---
        # Goal: Inpainted patch takes precedence in overlapping areas. Original content is used elsewhere.

        # Contribution from the original image:
        # Valid original pixels, excluding areas covered by the inpainted patch.
        original_content_contribution = base_img * base_mask.unsqueeze(-1) * \
                                          (1.0 - inpainted_patch_coverage_mask.unsqueeze(-1))
        
        # Contribution from the inpainted patch (reprojected as back_img):
        # Valid inpainted pixels, where the patch provides coverage.
        inpainted_patch_contribution = back_img * inpainted_patch_coverage_mask.unsqueeze(-1)

        # Combine:
        final_img = original_content_contribution + inpainted_patch_contribution

        # --- needs_inpaint_mask Derivation ---
        # Identifies areas that were initially holes AND remain un-filled by the reprojected inpainted patch.
        # These are areas that still require inpainting if a further pass was to be made.
        not_covered_by_inpainted_patch = 1.0 - inpainted_patch_coverage_mask
        needs_inpaint_mask = initial_hole_mask * not_covered_by_inpainted_patch # Element-wise multiplication (AND logic)

        return final_img, needs_inpaint_mask

# register
NODE_CLASS_MAPPINGS = {
    "OutpaintAnyProjection": OutpaintAnyProjection
}
