"""
xuat_tuyen_dm.py — BƯỚC 1/3: xuất tuyến DM + thầy từ `cap_nut` đã lưu sẵn trong .pkl.

`.pkl` của bão đã có `cap_nut[k][a]` = [tuyến_chính, dự phòng1, ...], mỗi tuyến là
DANH SÁCH NÚT. Nên chỉ cần đọc chỉ số lựa chọn (`a_dm` từ model, `nhan` của thầy)
rồi tra thẳng — KHÔNG dựng lại mạng (vài giây thay vì 11 phút).

Nếu .pkl chưa có `cap_nut`, chạy `python sinh_lai_capnut.py` trước.

    python xuat_tuyen_dm.py --tap soulik_2024_k4_5 --ten dm6 --out duong_di_nut.json
    python gop_tuyen.py --tap soulik_2024_k4_5
    python ban_do_3d.py --bao soulik
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parent
if str(GOC) not in sys.path:
    sys.path.insert(0, str(GOC))

RA = GOC / "data" / "bao_da_tuyen"


def nap_model(thu_muc: str, ten: str):
    from src.nhieu_luong import mo_hinh_dt as md
    return md.ham_chon(GOC / "results" / "bao_da_tuyen" / thu_muc, ten, softmax=False)["dm"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tap", default="soulik_2024_k4_5")
    ap.add_argument("--thu-muc", default="mo_hinh6")
    ap.add_argument("--ten", default="dm6")
    ap.add_argument("--out", default="duong_di_nut.json")
    a = ap.parse_args()

    tb = a.tap.split("_k")[0]
    ps = sorted((RA / a.tap).glob("kb_*.pkl"))
    if not ps:
        raise SystemExit(f"không có .pkl trong {RA / a.tap}")

    print(f"bão: {tb} | {len(ps)} tệp | model: {a.thu_muc}/{a.ten}.pt")
    model = nap_model(a.thu_muc, a.ten)

    ra, bo_qua = {}, 0
    t0 = time.time()
    for q, pk in enumerate(ps, 1):
        for r in pickle.load(open(pk, "rb")):
            cap = r.get("cap_nut")
            if not cap:
                bo_qua += 1
                continue
            a_dm = model(r["gon"])
            khoa = f"{pk.stem}|{int(r['t'])}|{int(r['kb'])}"
            item = {"t": int(r["t"]), "K": int(r["K"]), "nguon": list(r["nguon"]), "bao": tb,
                    "dm": [cap[k][int(a_dm[k])][0] for k in range(len(cap))]}
            if r.get("nhan") is not None:
                item["thay"] = [cap[k][int(r["nhan"][k])][0] for k in range(len(cap))]
            # dự phòng: tuyến thứ 2 của mỗi luồng (nếu có) — dùng cho `cap_phu`
            item["dm_phu"] = [cap[k][int(a_dm[k])][1] for k in range(len(cap))
                              if len(cap[k][int(a_dm[k])]) > 1]
            ra[khoa] = item
        print(f"  {q}/{len(ps)} {pk.name}", flush=True)

    out = GOC / a.out
    cu = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    cu.update(ra)
    out.write_text(json.dumps(cu, separators=(",", ":")), encoding="utf-8")
    print(f"\nbỏ qua (không có cap_nut): {bo_qua}")
    print(f"đã ghi: {out} ({len(ra)} bản ghi, {time.time()-t0:.1f}s)")

    k0 = next(iter(ra))
    print(f"\nví dụ {k0}:")
    print(f"  nguồn: {ra[k0]['nguon']}")
    print(f"  DM luồng 0: {ra[k0]['dm'][0]}")
    if "thay" in ra[k0]:
        print(f"  thầy luồng 0: {ra[k0]['thay'][0]}")


if __name__ == "__main__":
    main()
