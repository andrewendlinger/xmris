import dataclasses
import shutil
import textwrap
from collections import defaultdict
from dataclasses import field
from typing import Any, NamedTuple


def PlotParam(default: Any, group: str, description: str) -> Any:
    """Wrap ``dataclasses.field`` to attach *group* and *description* metadata."""
    return field(default=default, metadata={"group": group, "description": description})


class _FieldInfo(NamedTuple):
    """Resolved display information for a single config field."""

    name: str
    value: str
    type_name: str
    description: str


class BasePlotConfig:
    """Base configuration class for xmris plotting.

    Provides rich display methods for Jupyter Notebooks and terminal
    environments.  Inherit from this class when creating new configs.
    """

    _DEFAULT_GROUP = "Other / Ungrouped"
    _COLUMNS = ("Parameter", "Current Value", "Type", "Description")

    _CSS = textwrap.dedent("""\
        <style>
            .rpc-container { font-family: sans-serif; max-width: 850px; line-height: 1.3; }
            .rpc-header    { margin: 0 0 4px 0; font-size: 16px; }
            .rpc-desc      { opacity: 0.8; margin: 0 0 8px 0; font-size: 12px; }
            .rpc-table     { width: 100%; border-collapse: collapse; text-align: left; font-size: 12px; }
            .rpc-table th,
            .rpc-table td  { padding: 4px 8px; }
            .rpc-th        { border-bottom: 2px solid rgba(128,128,128,.5);
                             background: rgba(128,128,128,.15); text-align: center; }
            .rpc-tr        { border-bottom: 1px solid rgba(128,128,128,.2); }
            .rpc-group     { font-weight: bold; background: rgba(128,128,128,.08);
                             text-align: right; font-size: 11px;
                             text-transform: uppercase; letter-spacing: .5px; }
            .rpc-type      { opacity: 0.7; font-style: italic; }
            .rpc-center    { text-align: center; }
        </style>
    """)

    # Inline fallback for renderers that strip <style> blocks (e.g. MyST).
    _GROUP_INLINE_STYLE = (
        "text-align:right; font-weight:bold; "
        "background:rgba(128,128,128,.08); text-transform:uppercase;"
    )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _type_name(f: dataclasses.Field) -> str:
        """Human-readable name for a field's type annotation."""
        return f.type.__name__ if hasattr(f.type, "__name__") else str(f.type)

    def _resolve_field(self, f: dataclasses.Field) -> _FieldInfo:
        """Bundle the display-relevant attributes of *f* into a `_FieldInfo`."""
        return _FieldInfo(
            name=f.name,
            value=repr(getattr(self, f.name)),
            type_name=self._type_name(f),
            description=f.metadata.get("description", ""),
        )

    def _grouped_fields(self) -> dict[str, list[dataclasses.Field]]:
        """Dataclass fields organised by their ``group`` metadata."""
        groups: dict[str, list[dataclasses.Field]] = defaultdict(list)
        for f in dataclasses.fields(self):
            groups[f.metadata.get("group", self._DEFAULT_GROUP)].append(f)
        return groups

    def _description(self) -> str:
        """First non-empty line of the class docstring, or a generic fallback."""
        if doc := self.__class__.__doc__:
            for line in doc.strip().splitlines():
                if stripped := line.strip():
                    return stripped
        return f"Current settings for {self.__class__.__name__}"

    @staticmethod
    def _wrap_description(text: str, width: int) -> list[str]:
        """Wrap a (possibly multi-line) description to fit *width* columns."""
        wrapped: list[str] = []
        for line in text.splitlines():
            wrapped.extend(textwrap.wrap(line, width=width) or [""])
        return wrapped or [""]

    # ------------------------------------------------------------------ #
    #  Rich representations                                                #
    # ------------------------------------------------------------------ #

    def _repr_html_(self) -> str:
        """Jupyter: theme-aware HTML table."""
        name = self.__class__.__name__
        header = "".join(f"<th class='rpc-th'>{c}</th>" for c in self._COLUMNS)

        body: list[str] = []
        for group, fields in self._grouped_fields().items():
            body.append(
                f"<tr><td colspan='4' class='rpc-group' "
                f"style='{self._GROUP_INLINE_STYLE}'>{group}</td></tr>"
            )
            for f in fields:
                info = self._resolve_field(f)
                body.append(
                    f"<tr class='rpc-tr'>"
                    f"<td><strong>{info.name}</strong></td>"
                    f"<td class='rpc-center'><code>{info.value}</code></td>"
                    f"<td class='rpc-center rpc-type'>{info.type_name}</td>"
                    f"<td>{info.description}</td>"
                    f"</tr>"
                )

        return (
            f"{self._CSS}"
            f"<div class='rpc-container'>"
            f"<h4 class='rpc-header'>{name}</h4>"
            f"<p class='rpc-desc'>{self._description()}</p>"
            f"<table class='rpc-table'><tr>{header}</tr>"
            f"{''.join(body)}"
            f"</table></div>"
        )

    def _repr_markdown_(self) -> str:
        """Markdown fallback (e.g. GitHub READMEs)."""
        name = self.__class__.__name__
        lines = [f"### {name}", f"\n*{self._description()}*\n"]

        for group, fields in self._grouped_fields().items():
            lines.append(
                f"\n<div style='text-align:right; font-weight:bold; "
                f"text-transform:uppercase;'>{group}</div>\n"
            )
            lines.append("| Parameter | Current Value | Type | Description |")
            lines.append("| :--- | :---: | :---: | :--- |")
            for f in fields:
                info = self._resolve_field(f)
                lines.append(
                    f"| `{info.name}` | `{info.value}` | *{info.type_name}* | {info.description} |"
                )

        return "\n".join(lines)

    def __str__(self) -> str:
        """Terminal-friendly text table with dynamic column widths."""
        term_width = shutil.get_terminal_size((100, 20)).columns
        infos = [self._resolve_field(f) for f in dataclasses.fields(self)]

        widths = (
            max(len(i.name) for i in infos) + 2,
            max(len(i.value) for i in infos) + 2,
            max(len(i.type_name) for i in infos) + 2,
        )
        desc_width = max(term_width - sum(widths) - 6, 20)

        sep = "=" * term_width
        lines = [
            f"\n{sep}",
            f"{self.__class__.__name__} - Current Settings".center(term_width),
            sep,
        ]

        for group, group_fields in self._grouped_fields().items():
            lines.append(f"\n[ {group.upper()} ]".rjust(term_width))

            for f in group_fields:
                info = self._resolve_field(f)
                wrapped = self._wrap_description(info.description, desc_width)

                lines.append(
                    f"  {info.name:<{widths[0]}}"
                    f" {info.value:<{widths[1]}}"
                    f" {info.type_name:<{widths[2]}}"
                    f" ┃ {wrapped[0]}"
                )
                padding = " " * (sum(widths) + 3)
                lines.extend(f"{padding}┃ {dl}" for dl in wrapped[1:])

        lines.append(f"{sep}\n")
        return "\n".join(lines)
