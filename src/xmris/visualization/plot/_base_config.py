import dataclasses
import shutil
import textwrap
from collections import defaultdict


class BasePlotConfig:
    """Base configuration class for xmris plotting.

    Provides rich display methods for Jupyter Notebooks and terminal environments.
    Inherit from this class when creating new configs.
    """

    def _get_grouped_fields(self):
        """Help to organize fields by group metadata."""
        grouped = defaultdict(list)
        for f in dataclasses.fields(self):
            group_name = f.metadata.get("group", "Other / Ungrouped")
            grouped[group_name].append(f)
        return grouped

    def _get_description(self) -> str:
        """Extract the first line of the class docstring to use as a description."""
        doc = self.__class__.__doc__
        if doc:
            lines = [line.strip() for line in doc.strip().split("\n") if line.strip()]
            if lines:
                return lines[0]
        return f"Current settings for {self.__class__.__name__}:"

    def _repr_html_(self) -> str:
        """Primary Jupyter output: beautifully formatted, theme-aware HTML table."""
        cls_name = self.__class__.__name__
        desc_text = self._get_description()

        css = """
        <style>
            .rpc-container { font-family: sans-serif; max-width: 850px; line-height: 1.3; }
            .rpc-header { margin: 0 0 4px 0; font-size: 16px; }
            .rpc-desc-text { opacity: 0.8; margin: 0 0 8px 0; font-size: 12px; }
            .rpc-table { width: 100%; border-collapse: collapse; text-align: left; font-size: 12px; }
            .rpc-table th, .rpc-table td { padding: 4px 8px; }

            .rpc-th {
                border-bottom: 2px solid rgba(128, 128, 128, 0.5);
                background-color: rgba(128, 128, 128, 0.15);
                text-align: center;
            }
            .rpc-tr { border-bottom: 1px solid rgba(128, 128, 128, 0.2); }
            .rpc-group {
                font-weight: bold;
                background-color: rgba(128, 128, 128, 0.08);
                text-align: right;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .rpc-type { opacity: 0.7; font-style: italic; }
            .rpc-center-col { text-align: center; }
        </style>
        """  # noqa: E501
        html = [
            css,
            "<div class='rpc-container'>",
            f"<h4 class='rpc-header'>{cls_name}</h4>",
            f"<p class='rpc-desc-text'>{desc_text}</p>",
            "<table class='rpc-table'>",
            "<tr>",
            "<th class='rpc-th'>Parameter</th>",
            "<th class='rpc-th'>Current Value</th>",
            "<th class='rpc-th'>Type</th>",
            "<th class='rpc-th'>Description</th>",
            "</tr>",
        ]

        for group_name, fields in self._get_grouped_fields().items():
            html.append("<tr>")
            html.append(f"<td colspan='4' class='rpc-group'>{group_name}</td>")
            html.append("</tr>")

            for f in fields:
                key = f.name
                val = getattr(self, key)
                val_type = f.type.__name__ if hasattr(f.type, "__name__") else str(f.type)
                desc = f.metadata.get("description", "")

                html.append("<tr class='rpc-tr'>")
                html.append(f"<td><strong>{key}</strong></td>")
                html.append(f"<td class='rpc-center-col'><code>{val!r}</code></td>")
                html.append(f"<td class='rpc-center-col rpc-type'>{val_type}</td>")
                html.append(f"<td>{desc}</td>")
                html.append("</tr>")

        html.append("</table></div>")
        return "".join(html)

    def _repr_markdown_(self) -> str:
        """Fallback for Markdown-only renderers (e.g., GitHub READMEs)."""
        cls_name = self.__class__.__name__
        desc_text = self._get_description()

        md = [f"### {cls_name}", f"\n*{desc_text}*\n"]

        for group_name, fields in self._get_grouped_fields().items():
            md.append(f"\n**{group_name}**\n")
            md.append("| Parameter | Current Value | Type | Description |")
            md.append("| :--- | :---: | :---: | :--- |")
            for f in fields:
                key = f.name
                val = getattr(self, key)
                val_type = f.type.__name__ if hasattr(f.type, "__name__") else str(f.type)
                desc = f.metadata.get("description", "")
                md.append(f"| `{key}` | `{val!r}` | *{val_type}* | {desc} |")

        return "\n".join(md)

    def __str__(self) -> str:
        """Terminal text fallback. Dynamically aligns columns and wraps text."""
        term_width = shutil.get_terminal_size((100, 20)).columns

        fields = dataclasses.fields(self)
        col1_w = max(len(f.name) for f in fields) + 2
        col2_w = max(len(repr(getattr(self, f.name))) for f in fields) + 2
        col3_w = (
            max(
                len(f.type.__name__ if hasattr(f.type, "__name__") else str(f.type))
                for f in fields
            )
            + 2
        )

        col4_w = term_width - col1_w - col2_w - col3_w - 6
        col4_w = max(col4_w, 20)

        cls_name = self.__class__.__name__
        lines = ["\n" + "=" * term_width]
        lines.append(f"{cls_name} - Current Settings".center(term_width))
        lines.append("=" * term_width)

        for group_name, group_fields in self._get_grouped_fields().items():
            lines.append(f"\n[ {group_name.upper()} ]")

            for f in group_fields:
                key = f.name
                val = repr(getattr(self, key))
                val_type = f.type.__name__ if hasattr(f.type, "__name__") else str(f.type)
                desc = f.metadata.get("description", "")

                wrapped_desc = []
                for line in desc.splitlines():
                    wrapped_desc.extend(textwrap.wrap(line, width=col4_w) or [""])
                if not wrapped_desc:
                    wrapped_desc = [""]

                lines.append(
                    f"  {key:<{col1_w}}"
                    + f" {val:<{col2_w}}"
                    + f" {val_type:<{col3_w}}"
                    + f" ┃ {wrapped_desc[0]}"
                )

                empty_space = " " * (col1_w + col2_w + col3_w + 3)
                for desc_line in wrapped_desc[1:]:
                    lines.append(f"{empty_space}┃ {desc_line}")

        lines.append("=" * term_width + "\n")
        return "\n".join(lines)
