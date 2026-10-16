"""Tiny, dependency-free builders for Superset v1 import bundles (databases / datasets / charts /
dashboards). Files are written as JSON, which YAML parsers (Superset uses PyYAML) read natively.

Only the stable core of each chart's ``params`` is set; Superset fills UI defaults for anything
omitted. UUIDs are deterministic (uuid5), so re-imports update objects in place.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

NAMESPACE = uuid.UUID("6f3c2a7e-3b7d-4f7a-9a51-0c2d8b1e4a10")
BUNDLE_VERSION = "1.0.0"

POSITION_COLORS = {"DB": "#2a78d6", "OL": "#eb6834", "DL": "#1baf7a", "WR": "#eda100", "TE": "#e87ba4",
                   "UNKNOWN": "#898781"}
STATUS_COLORS = {"PASS": "#0ca30c", "WARN": "#fab219", "FAIL": "#d03b3b", "ERROR": "#d03b3b",
                 "1 CRITICAL": "#d03b3b", "2 HIGH": "#ec835a", "3 MEDIUM": "#fab219", "4 LOW": "#898781",
                 "5 INFO": "#c3c2b7", "OK": "#0ca30c", "OK_WITH_WARNINGS": "#fab219", "FAILED": "#d03b3b"}
LABEL_COLORS = {**POSITION_COLORS, **STATUS_COLORS}


def uid(kind: str, key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"nfl-lab/{kind}/{key}"))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# ------------------------------------------------------------------------------- metrics
def M(sql: str, label: str) -> dict:
    """Ad-hoc SQL metric."""
    return {"expressionType": "SQL", "sqlExpression": sql, "label": label, "hasCustomLabel": True,
            "optionName": f"metric_{hashlib.md5((sql + label).encode()).hexdigest()[:12]}"}


def W(sql: str) -> dict:
    """Ad-hoc SQL WHERE filter fixed on a chart."""
    return {"expressionType": "SQL", "sqlExpression": sql, "clause": "WHERE",
            "filterOptionName": f"filter_{hashlib.md5(sql.encode()).hexdigest()[:12]}"}


COUNT = M("COUNT(*)", "Rows")


# ------------------------------------------------------------------------------- objects
@dataclass
class Dataset:
    key: str
    table: str
    description: str
    columns: list[str]            # "name:TYPE"; TIMESTAMP/DATE columns become temporal
    schema: str = "lab"
    dttm: str | None = None

    @property
    def uuid(self) -> str:
        return uid("dataset", self.key)


@dataclass
class Chart:
    key: str
    title: str
    dataset: Dataset
    params: dict
    description: str = ""

    @property
    def uuid(self) -> str:
        return uid("chart", self.key)

    @property
    def viz_type(self) -> str:
        return self.params["viz_type"]


@dataclass
class Filter:
    name: str
    dataset: Dataset
    column: str
    multi: bool = True
    parents: tuple[str, ...] = ()        # names of parent filters (cascade)
    description: str = ""

    def fid(self, dashboard: str) -> str:
        return "NATIVE_FILTER-" + uuid.uuid5(NAMESPACE, f"{dashboard}/{self.name}").hex[:10]


@dataclass
class Dashboard:
    key: str
    title: str
    slug: str
    rows: list[list[tuple]]              # items: ("chart", Chart, width, height) | ("md", text, width, height)
    filters: list[Filter] = field(default_factory=list)
    description: str = ""

    @property
    def uuid(self) -> str:
        return uid("dashboard", self.key)

    def charts(self) -> list[Chart]:
        seen, out = set(), []
        for row in self.rows:
            for item in row:
                if item[0] == "chart" and item[1].key not in seen:
                    seen.add(item[1].key)
                    out.append(item[1])
        return out


# ------------------------------------------------------------------------------- chart builders
def _base(viz: str, row_limit: int, filters=()) -> dict:
    return {"viz_type": viz, "adhoc_filters": list(filters), "row_limit": row_limit, "color_scheme": "nflLab"}


def big_number(key, title, ds, metric, subheader="", fmt="SMART_NUMBER", filters=(), description="") -> Chart:
    p = {**_base("big_number_total", 1, filters), "metric": metric, "subheader": subheader,
         "header_font_size": 0.3, "subheader_font_size": 0.125, "y_axis_format": fmt, "time_format": "smart_date"}
    return Chart(key, title, ds, p, description)


def bar(key, title, ds, x, metrics, groupby=(), stack=False, horizontal=False, fmt="SMART_NUMBER",
        row_limit=1000, sort_by=None, sort_asc=True, categorical=True, filters=(), x_title="", y_title="",
        description="") -> Chart:
    p = {**_base("echarts_timeseries_bar", row_limit, filters), "x_axis": x, "metrics": list(metrics),
         "groupby": list(groupby), "orientation": "horizontal" if horizontal else "vertical",
         "stack": "Stack" if stack else None, "xAxisForceCategorical": categorical,
         "x_axis_sort": sort_by or x, "x_axis_sort_asc": sort_asc, "show_legend": True, "legendType": "scroll",
         "legendOrientation": "top", "rich_tooltip": True, "y_axis_format": fmt, "truncateYAxis": False,
         "x_axis_title": x_title, "x_axis_title_margin": 30 if x_title else 0, "y_axis_title": y_title,
         "y_axis_title_margin": 40 if y_title else 0, "y_axis_title_position": "Left", "show_value": False,
         "only_total": True, "zoomable": False}
    return Chart(key, title, ds, p, description)


def line(key, title, ds, x, metrics, groupby=(), time_grain=None, fmt="SMART_NUMBER", row_limit=10000,
         filters=(), description="") -> Chart:
    p = {**_base("echarts_timeseries_line", row_limit, filters), "x_axis": x, "time_grain_sqla": time_grain,
         "metrics": list(metrics), "groupby": list(groupby), "markerEnabled": True, "markerSize": 6,
         "show_legend": True, "legendType": "scroll", "rich_tooltip": True, "y_axis_format": fmt}
    return Chart(key, title, ds, p, description)


def pie(key, title, ds, groupby, metric, filters=(), description="") -> Chart:
    p = {**_base("pie", 100, filters), "groupby": list(groupby), "metric": metric, "donut": True,
         "innerRadius": 40, "outerRadius": 70, "show_labels": True, "label_type": "key_percent",
         "labels_outside": True, "show_legend": True, "sort_by_metric": True, "number_format": "SMART_NUMBER"}
    return Chart(key, title, ds, p, description)


def table_raw(key, title, ds, columns, order_by=None, desc=True, row_limit=1000, page_length=25, filters=(),
              description="") -> Chart:
    p = {**_base("table", row_limit, filters), "query_mode": "raw", "all_columns": list(columns),
         "order_by_cols": [f'["{order_by}", {str(not desc).lower()}]'] if order_by else [],
         "server_pagination": False, "page_length": page_length, "include_search": True,
         "table_timestamp_format": "smart_date", "allow_render_html": False, "show_cell_bars": False}
    return Chart(key, title, ds, p, description)


def table_agg(key, title, ds, groupby, metrics, row_limit=1000, page_length=25, filters=(), description="") -> Chart:
    p = {**_base("table", row_limit, filters), "query_mode": "aggregate", "groupby": list(groupby),
         "metrics": list(metrics), "timeseries_limit_metric": metrics[0], "order_desc": True,
         "percent_metrics": [], "include_search": True, "page_length": page_length, "show_cell_bars": True,
         "allow_render_html": False}
    return Chart(key, title, ds, p, description)


def heatmap(key, title, ds, x, y, metric, fmt=".0%", row_limit=10000, filters=(), description="") -> Chart:
    p = {**_base("heatmap_v2", row_limit, filters), "x_axis": x, "groupby": y, "metric": metric,
         "linear_color_scheme": "superset_seq_1", "normalize_across": "heatmap", "show_legend": True,
         "show_values": True, "show_percentage": False, "value_bounds": [None, None], "y_axis_format": fmt,
         "sort_x_axis": "alpha_asc", "sort_y_axis": "alpha_asc", "xscale_interval": -1, "yscale_interval": -1,
         "left_margin": "auto", "bottom_margin": "auto"}
    return Chart(key, title, ds, p, description)


def histogram(key, title, ds, column, groupby=(), bins=25, row_limit=10000, filters=(), x_title="",
              description="") -> Chart:
    p = {**_base("histogram_v2", row_limit, filters), "column": column, "groupby": list(groupby), "bins": bins,
         "normalize": False, "cumulative": False, "show_legend": True, "show_value": False,
         "x_axis_title": x_title, "y_axis_title": "count"}
    return Chart(key, title, ds, p, description)


def box(key, title, ds, series, distribute_across, metric, fmt="SMART_NUMBER", row_limit=10000, filters=(),
        description="") -> Chart:
    p = {**_base("box_plot", row_limit, filters), "groupby": list(series), "columns": list(distribute_across),
         "metrics": [metric], "whiskerOptions": "Tukey", "number_format": fmt, "date_format": "smart_date"}
    return Chart(key, title, ds, p, description)


def bubble(key, title, ds, entity, x, y, size, series, x_label="", y_label="", row_limit=2000, filters=(),
           description="") -> Chart:
    p = {**_base("bubble_v2", row_limit, filters), "entity": entity, "x": x, "y": y, "size": size,
         "series": series, "max_bubble_size": "15", "opacity": 0.6, "show_legend": True,
         "x_axis_label": x_label, "y_axis_label": y_label, "xAxisFormat": "SMART_NUMBER",
         "y_axis_format": "SMART_NUMBER", "logXAxis": False, "logYAxis": False, "tooltipSizeFormat": "SMART_NUMBER"}
    return Chart(key, title, ds, p, description)


# ------------------------------------------------------------------------------- serialisation
def _col(spec: str, dttm: str | None) -> dict:
    name, _, typ = spec.partition(":")
    typ = typ or "VARCHAR"
    return {"column_name": name, "verbose_name": None, "is_dttm": typ in ("TIMESTAMP", "DATE") or name == dttm,
            "is_active": True, "type": typ, "groupby": True, "filterable": True, "expression": None,
            "description": None, "python_date_format": None, "extra": None}


# Superset 5.0 `import_database` does `config.pop("allow_csv_upload")` with no default → KeyError when
# absent. These keys mirror a Superset 5.0 database export; build_bundle.check_bundle() enforces them.
DATABASE_REQUIRED_KEYS = ("database_name", "sqlalchemy_uri", "cache_timeout", "expose_in_sqllab", "allow_run_async",
                          "allow_ctas", "allow_cvas", "allow_dml", "allow_csv_upload", "extra", "uuid", "version")
DUCKDB_QUERY_MEMORY = "512MB"   # per Superset query connection; keeps worker RSS well under SUPERSET_MEM_LIMIT


def database_yaml(name: str, uri: str, key: str = "nfl_lab_duckdb") -> dict:
    return {
        "database_name": name, "sqlalchemy_uri": uri, "cache_timeout": None, "expose_in_sqllab": True,
        "allow_run_async": False, "allow_ctas": False, "allow_cvas": False, "allow_dml": False,
        "allow_csv_upload": False,
        "extra": {
            "engine_params": {"connect_args": {
                "read_only": True,
                # Snapshot is self-contained: forbid reading any other file / installing extensions.
                "config": {"enable_external_access": False, "memory_limit": DUCKDB_QUERY_MEMORY, "threads": 2},
            }},
            "metadata_params": {}, "schemas_allowed_for_file_upload": [], "allows_virtual_table_explore": True,
        },
        "uuid": uid("database", key), "version": BUNDLE_VERSION,
    }


def dataset_yaml(ds: Dataset, database_uuid: str) -> dict:
    return {
        "table_name": ds.table, "main_dttm_col": ds.dttm, "description": ds.description, "default_endpoint": None,
        "offset": 0, "cache_timeout": None, "schema": ds.schema, "sql": None, "params": None,
        "template_params": None, "filter_select_enabled": True, "fetch_values_predicate": None, "extra": None,
        "uuid": ds.uuid,
        "metrics": [{"metric_name": "count", "verbose_name": "COUNT(*)", "metric_type": "count",
                     "expression": "COUNT(*)", "description": None, "d3format": None, "extra": None,
                     "warning_text": None}],
        "columns": [_col(c, ds.dttm) for c in ds.columns], "version": BUNDLE_VERSION, "database_uuid": database_uuid,
    }


def chart_yaml(c: Chart) -> dict:
    return {"slice_name": c.title, "description": c.description or None, "certified_by": None,
            "certification_details": None, "viz_type": c.viz_type, "params": c.params, "query_context": None,
            "cache_timeout": None, "uuid": c.uuid, "version": BUNDLE_VERSION, "dataset_uuid": c.dataset.uuid}


def dashboard_yaml(d: Dashboard) -> dict:
    position: dict[str, Any] = {
        "DASHBOARD_VERSION_KEY": "v2",
        "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
        "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": [], "parents": ["ROOT_ID"]},
        "HEADER_ID": {"id": "HEADER_ID", "type": "HEADER", "meta": {"text": d.title}},
    }
    for r, row in enumerate(d.rows):
        row_id = f"ROW-{d.key}-{r}"
        position["GRID_ID"]["children"].append(row_id)
        position[row_id] = {"type": "ROW", "id": row_id, "children": [], "parents": ["ROOT_ID", "GRID_ID"],
                            "meta": {"background": "BACKGROUND_TRANSPARENT"}}
        for i, item in enumerate(row):
            kind, obj, width, height = item
            if kind == "chart":
                cid = f"CHART-{obj.key}"[:60]
                position[cid] = {"type": "CHART", "id": cid, "children": [], "parents": ["ROOT_ID", "GRID_ID", row_id],
                                 "meta": {"width": width, "height": height, "chartId": 0, "uuid": obj.uuid,
                                          "sliceName": obj.title}}
            else:
                cid = f"MARKDOWN-{d.key}-{r}-{i}"
                position[cid] = {"type": "MARKDOWN", "id": cid, "children": [], "parents": ["ROOT_ID", "GRID_ID", row_id],
                                 "meta": {"width": width, "height": height, "code": obj}}
            position[row_id]["children"].append(cid)
    by_name = {f.name: f for f in d.filters}
    native = [{
        "id": f.fid(d.key), "name": f.name, "filterType": "filter_select", "type": "NATIVE_FILTER",
        "targets": [{"datasetUuid": f.dataset.uuid, "column": {"name": f.column}}],
        "controlValues": {"enableEmptyFilter": False, "defaultToFirstItem": False, "multiSelect": f.multi,
                          "searchAllOptions": True, "inverseSelection": False},
        "defaultDataMask": {"extraFormData": {}, "filterState": {}, "ownState": {}},
        "cascadeParentIds": [by_name[p].fid(d.key) for p in f.parents],
        "scope": {"rootPath": ["ROOT_ID"], "excluded": []},
        "description": f.description, "chartsInScope": [], "tabsInScope": [],
    } for f in d.filters]
    return {
        "dashboard_title": d.title, "description": d.description or None, "css": "", "slug": d.slug,
        "certified_by": None, "certification_details": None, "published": True, "uuid": d.uuid,
        "position": position,
        "metadata": {"color_scheme": "nflLab", "label_colors": LABEL_COLORS, "refresh_frequency": 0,
                     "expanded_slices": {}, "timed_refresh_immune_slices": [], "cross_filters_enabled": True,
                     "native_filter_configuration": native},
        "version": BUNDLE_VERSION,
    }
