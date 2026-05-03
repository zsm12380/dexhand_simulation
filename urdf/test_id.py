import mujoco
import numpy as np


MODEL_PATH = "dexhand_lh_rl.xml"


def safe_name(model, objtype, objid):
    name = mujoco.mj_id2name(model, objtype, objid)
    return name if name is not None else "None"


def print_bodies(model):
    print("\n========== BODIES ==========")
    for i in range(model.nbody):
        print(f"body_id={i:3d}, body_name={safe_name(model, mujoco.mjtObj.mjOBJ_BODY, i)}")


def print_geoms(model):
    print("\n========== GEOMS ==========")
    for gid in range(model.ngeom):
        gname = safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        bid = model.geom_bodyid[gid]
        bname = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)

        gtype = model.geom_type[gid]
        contype = model.geom_contype[gid]
        conaffinity = model.geom_conaffinity[gid]

        print(
            f"geom_id={gid:3d}, geom_name={gname:10s}, body_id={bid:3d}, body_name={bname:20s}, "
            f"type={gtype}, contype={contype}, conaffinity={conaffinity}"
        )


def print_sites(model):
    print("\n========== SITES ==========")
    for sid in range(model.nsite):
        sname = safe_name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        bid = model.site_bodyid[sid]
        bname = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"site_id={sid:3d}, site_name={sname:20s}, body_id={bid:3d}, body_name={bname}")


def nearest_geoms_to_sites(model, data, site_names, topk=10):
    print("\n========== NEAREST GEOMS TO SITES ==========")

    # forward 一次，拿到 site/geom 世界坐标
    mujoco.mj_forward(model, data)

    geom_positions = []
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        gpos = data.geom_xpos[gid].copy()
        gname = safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        bname = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        geom_positions.append((gid, gpos, gname, bname))

    for site_name in site_names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            print(f"[WARN] site not found: {site_name}")
            continue

        spos = data.site_xpos[sid].copy()
        sbid = model.site_bodyid[sid]
        sbname = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, sbid)

        dists = []
        for gid, gpos, gname, bname in geom_positions:
            dist = np.linalg.norm(spos - gpos)
            dists.append((dist, gid, gname, bname))

        dists.sort(key=lambda x: x[0])

        print(f"\n--- site: {site_name}, body={sbname} ---")
        for rank, (dist, gid, gname, bname) in enumerate(dists[:topk]):
            print(
                f"rank={rank:2d}, dist={dist:.6f}, geom_id={gid:3d}, geom_name={gname:10s}, body_name={bname}"
            )


def print_body_geom_mapping(model):
    print("\n========== BODY -> GEOMS ==========")
    body_to_geoms = {}
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        body_to_geoms.setdefault(bid, []).append(gid)

    for bid in range(model.nbody):
        bname = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        gids = body_to_geoms.get(bid, [])
        if len(gids) == 0:
            continue
        print(f"body_id={bid:3d}, body_name={bname:20s}, geoms={gids}")


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print_bodies(model)
    print_geoms(model)
    print_sites(model)
    print_body_geom_mapping(model)

    nearest_geoms_to_sites(
        model,
        data,
        site_names=["th_tip_site", "ff_tip_site", "mf_tip_site", "rf_tip_site", "lf_tip_site"],
        topk=10,
    )


if __name__ == "__main__":
    main()
