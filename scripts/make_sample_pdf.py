"""Create documents/sample.pdf so the pipeline can be tried without your own file."""
from pathlib import Path

import pymupdf

PAGES = [
    ("Deployment guide: overview",
     "This sample guide describes a small deployment. Step one is to build the container image. "
     "Step two is to push the image to a registry. Step three is to apply the Kubernetes manifests. "
     "Always verify the rollout status before sending traffic to the new version. " * 6),
    ("Authentication process",
     "Users authenticate with a short lived token issued by the identity provider. "
     "The API gateway validates the token signature and expiry on every request. "
     "Expired tokens are rejected and the client must request a new one. " * 7),
    ("Troubleshooting",
     "If a pod stays in Pending, check node resources and scheduling events. "
     "If a pod restarts repeatedly, read the previous container logs. "
     "If a service is unreachable, confirm the selector labels match the pods. " * 7),
]


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "documents" / "sample.pdf"
    out.parent.mkdir(exist_ok=True)
    doc = pymupdf.open()
    for title, body in PAGES:
        page = doc.new_page()
        page.insert_text((72, 72), title, fontsize=16)
        page.insert_textbox(pymupdf.Rect(72, 100, 523, 770), body, fontsize=11)
    doc.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
