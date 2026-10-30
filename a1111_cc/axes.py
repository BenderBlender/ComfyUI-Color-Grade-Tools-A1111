"""The decoder's measured colour axes, for the models they have been measured on.

``latent_rgb_factors`` is a least-squares fit of the decoded *picture* to the
latent, made for drawing previews. Fitted on pictures it is dominated by
brightness, and its three colour columns come out nearly dependent: for SDXL
the matrix has condition number 11, and the minimum-norm latent step that
should turn the projection one unit redder is 9.7 long, almost all of it along
the second and third latent channels — ten times the step that changes
brightness by the same amount. Along those channels the real decoder does not
answer the way the fit says it does (the fit's and the decoder's colour planes
are 10.6 degrees apart), so the big step decodes as red MINUS green, darker,
three to five times stronger than asked: that was the magenta the latent wheels
gave on SDXL (DEMO_RESULTS/fable/REPORT.md, 30.10).

What is needed for an *edit* is not the fit of the picture but the decoder's
response to a change: ``J = d(decoded RGB) / d(latent)``, measured by decoding
a latent nudged along each channel (``tools/calibrate_axes.py``). Its condition
number is 3.8 and a unit of red through it decodes as red (+0.035, +0.000,
+0.000 for 0.034 asked). Measured on the same model, ``tone`` is how much of a
latent colour change the decoder lets through at each brightness of the
picture: about 0.45 near black, 1.4 in the midtones, 0.5 near white. Both are
in the units of the projection's ``to_unit`` range.

Entries are keyed by the latent format's class name. A family without an entry
falls back to the preview matrix, which is right in shape and wrong in colour
by the amounts above.
"""

from typing import Optional

MEASURED = {
    # ProtoVision XL (the SDXL VAE), two 1024^2 pictures, central differences of +-0.05 in model space; the bias
    # puts the projection's mean on the decode's mean. DEMO_RESULTS/logs/exp_jacobian.json, fable/f4_tone_response.json.
    "SDXL": {
        "factors": [[0.3606, 0.4025, 0.4282],
                    [-0.2189, 0.1009, 0.1471],
                    [0.1504, 0.2691, -0.1126],
                    [-0.4377, -0.2758, -0.2815]],
        "bias": [0.1013, -0.0238, -0.1087],
        "tone": {
            "v": [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475,
                  0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975],
            "r": [0.44, 0.63, 0.81, 0.98, 1.11, 1.21, 1.30, 1.37, 1.40, 1.42,
                  1.44, 1.44, 1.49, 1.52, 1.47, 1.29, 1.14, 0.92, 0.71, 0.52],
        },
    },
    # SD3.5 medium (the SD3 VAE, 16 channels), measured 30.10 with tools/calibrate_axes.py on an SD3.5 dusk street and
    # the bright robot picture; J condition 2.7 / 3.0, +R through pinv(J) decoded +0.039 / +0.034 for 0.034 asked.
    # Through the preview matrix the wheels landed 24.5 degrees off the image grade on SD3.5.
    "SD3": {
        "factors": [[-0.0449, -0.0053, 0.0660],
                    [-0.0196, 0.0192, 0.0422],
                    [0.1078, 0.0188, -0.0354],
                    [0.1281, 0.0696, 0.1081],
                    [0.0550, 0.0213, -0.0478],
                    [0.0370, 0.1178, 0.0236],
                    [0.1083, 0.0159, 0.0199],
                    [-0.0296, 0.0608, 0.0988],
                    [-0.0867, -0.0858, -0.0121],
                    [-0.0323, 0.0143, -0.0271],
                    [0.0382, 0.0594, 0.0596],
                    [0.0172, 0.0633, -0.0141],
                    [0.0160, 0.0416, 0.0749],
                    [-0.0346, -0.0393, -0.1032],
                    [-0.0612, -0.0530, -0.0503],
                    [-0.0866, -0.0780, -0.0437]],
        "bias": [0.0621, 0.0212, -0.0386],
        "tone": {
            "v": [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975],
            "r": [0.50, 0.73, 0.92, 1.08, 1.20, 1.28, 1.34, 1.40, 1.42, 1.41, 1.39, 1.39, 1.39, 1.20, 1.07, 1.06, 0.98, 0.82, 0.63, 0.63],
        },
    },
}


def lookup(latent_format_name: str) -> Optional[dict]:
    """The measured axes of a latent format class, or None."""
    return MEASURED.get(latent_format_name)
