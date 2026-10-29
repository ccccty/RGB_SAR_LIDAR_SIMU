"""Material library: optical / thermal / RF properties (approximate, tunable).

rf_rho        : specular reflectivity at the radar band (0..1)
rough_m       : surface height roughness in metres (drives the Kirchhoff falloff)
emissivity    : broadband thermal emissivity, used for the heat balance (8-14 um)
emissivity_mwir: 3-5 um emissivity  (defaults to emissivity if absent)
emissivity_lwir: 8-12 um emissivity (defaults to emissivity if absent)
alpha_solar   : solar absorptivity
albedo        : visible RGB 0..1
h_int         : internal heat flux W/m^2 (engines, electronics, bodies)

Values are literature ranges for engineering estimates, not measurements.
Metal is the case where the band really matters: polished aluminium is a poor
emitter everywhere but worse in MWIR, so a bare-metal target looks colder in a
3-5 um image than in 8-12 um at the same physical temperature.
"""
MATERIALS = {
    "painted_metal": dict(rf_rho=0.55, rough_m=0.0020, emissivity=0.92,
                          emissivity_mwir=0.85, emissivity_lwir=0.92,
                          alpha_solar=0.35, albedo=(0.22, 0.24, 0.28), h_int=0.0),
    "metal_bare": dict(rf_rho=0.85, rough_m=0.0010, emissivity=0.12,
                       emissivity_mwir=0.08, emissivity_lwir=0.12,
                       alpha_solar=0.55, albedo=(0.62, 0.64, 0.66), h_int=0.0),
    "asphalt": dict(rf_rho=0.25, rough_m=0.0300, emissivity=0.95,
                    emissivity_mwir=0.93, emissivity_lwir=0.95,
                    alpha_solar=0.88, albedo=(0.16, 0.16, 0.17), h_int=0.0),
    "concrete": dict(rf_rho=0.35, rough_m=0.0150, emissivity=0.93,
                     emissivity_mwir=0.90, emissivity_lwir=0.93,
                     alpha_solar=0.65, albedo=(0.45, 0.44, 0.42), h_int=0.0),
    "foliage": dict(rf_rho=0.20, rough_m=0.0600, emissivity=0.97,
                    emissivity_mwir=0.95, emissivity_lwir=0.97,
                    alpha_solar=0.72, albedo=(0.16, 0.28, 0.12), h_int=0.0),
    "glass": dict(rf_rho=0.45, rough_m=0.0010, emissivity=0.85,
                  emissivity_mwir=0.70, emissivity_lwir=0.85,
                  alpha_solar=0.30, albedo=(0.18, 0.22, 0.26), h_int=0.0),
    "rubber": dict(rf_rho=0.20, rough_m=0.0080, emissivity=0.93,
                   emissivity_mwir=0.92, emissivity_lwir=0.93,
                   alpha_solar=0.92, albedo=(0.08, 0.08, 0.09), h_int=0.0),
    "clothing": dict(rf_rho=0.15, rough_m=0.0200, emissivity=0.96,
                     emissivity_mwir=0.90, emissivity_lwir=0.96,
                     alpha_solar=0.75, albedo=(0.30, 0.32, 0.38), h_int=0.0),
    "skin": dict(rf_rho=0.20, rough_m=0.0050, emissivity=0.98,
                 emissivity_mwir=0.97, emissivity_lwir=0.98,
                 alpha_solar=0.60, albedo=(0.62, 0.48, 0.40), h_int=0.0),
    "engine_hot": dict(rf_rho=0.60, rough_m=0.0020, emissivity=0.90,
                       emissivity_mwir=0.85, emissivity_lwir=0.90,
                       alpha_solar=0.40, albedo=(0.20, 0.20, 0.22), h_int=900.0),
    "seawater": dict(rf_rho=0.35, rough_m=0.0500, emissivity=0.96,
                     emissivity_mwir=0.96, emissivity_lwir=0.96,
                     alpha_solar=0.94, albedo=(0.04, 0.07, 0.10), h_int=0.0),
    "building_wall": dict(rf_rho=0.30, rough_m=0.0120, emissivity=0.90,
                          emissivity_mwir=0.88, emissivity_lwir=0.90,
                          alpha_solar=0.60, albedo=(0.55, 0.52, 0.48), h_int=0.0),
}


def eps(spec, band_name=None):
    """Emissivity of a material spec for a given IR band name (mwir/lwir/None)."""
    if band_name:
        v = spec.get("emissivity_" + band_name)
        if v is not None:
            return float(v)
    return float(spec.get("emissivity", 0.9))


def get(name):
    if name not in MATERIALS:
        raise KeyError("unknown material: %s (have: %s)" % (name, ", ".join(sorted(MATERIALS))))
    return MATERIALS[name]