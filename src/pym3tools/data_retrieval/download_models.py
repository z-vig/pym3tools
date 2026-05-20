from typing import Protocol, ClassVar, Self
from pathlib import Path, PurePosixPath
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse


class URLPath:
    def __init__(self, url: str):
        self._parsed = urlparse(url)

    def _replace_path(self, new_path: PurePosixPath):
        return URLPath(urlunparse(self._parsed._replace(path=str(new_path))))

    def joinpath(self, *parts) -> "URLPath":
        base = PurePosixPath(self._parsed.path)
        addition = PurePosixPath(*(Path(p).as_posix() for p in parts))
        return self._replace_path(base / addition)

    def with_suffix(self, suffix: str) -> "URLPath":
        path = PurePosixPath(self._parsed.path)
        return self._replace_path(path.with_suffix(suffix))

    def __truediv__(self, other):
        return self.joinpath(other)

    def __str__(self):
        return urlunparse(self._parsed)


class Downloadable(Protocol):
    def to_save(self) -> dict[str, Path]: ...


@dataclass
class DownloadPath:
    source: str
    target: Path

    def to_save(self) -> dict[str, Path]:
        return {str(self.source): self.target}


@dataclass
class ImageDownload:
    hdr: DownloadPath
    img: DownloadPath
    hdr_ext: ClassVar[str] = ".hdr"
    img_ext: ClassVar[str] = ".img"

    @classmethod
    def from_base(cls, src_base: URLPath, trg_base: Path) -> Self:
        return cls(
            hdr=DownloadPath(
                str(src_base.with_suffix(cls.hdr_ext.upper())),
                trg_base.with_suffix(cls.hdr_ext.lower()),
            ),
            img=DownloadPath(
                str(src_base.with_suffix(cls.img_ext.upper())),
                trg_base.with_suffix(cls.img_ext.lower()),
            ),
        )

    def to_save(self) -> dict[str, Path]:
        return {
            str(self.hdr.source): self.hdr.target,
            str(self.img.source): self.img.target,
        }


@dataclass
class TabDownload:
    tab: DownloadPath
    lbl: DownloadPath
    tab_ext: ClassVar[str] = ".tab"
    lbl_ext: ClassVar[str] = ".lbl"

    @classmethod
    def from_base(cls, src_base: URLPath, trg_base: Path) -> Self:
        return cls(
            tab=DownloadPath(
                str(src_base.with_suffix(cls.tab_ext.upper())),
                trg_base.with_suffix(cls.tab_ext.lower()),
            ),
            lbl=DownloadPath(
                str(src_base.with_suffix(cls.lbl_ext.upper())),
                trg_base.with_suffix(cls.lbl_ext.lower()),
            ),
        )

    @classmethod
    def from_base_nolbl(cls, src_base: URLPath, trg_base: Path) -> Self:
        return cls(
            tab=DownloadPath(
                str(src_base.with_suffix(cls.tab_ext.upper())),
                trg_base.with_suffix(cls.tab_ext.lower()),
            ),
            lbl=DownloadPath(
                "",
                trg_base.with_suffix(cls.lbl_ext.lower()),
            ),
        )

    def to_save(self) -> dict[str, Path]:
        return {
            self.tab.source: self.tab.target,
            self.lbl.source: self.lbl.target,
        }
