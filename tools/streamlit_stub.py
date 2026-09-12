"""Minimal stand-in for the `streamlit` API used by dashboard/app.py, for headless smoke tests.
Widgets return configurable values (defaults otherwise); outputs are recorded, and every
matplotlib figure / dataframe passed to Streamlit is validated."""
import types
from pathlib import Path

import pandas as pd


class StopApp(Exception):
    pass


class _Ctx:
    def __init__(self, st):
        self.st = st

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self.st, name)


def make_stub(widget_values: dict):
    st = types.ModuleType("streamlit")
    st.log = []
    st.errors = []

    def rec(kind, *a, **k):
        st.log.append((kind, a[0] if a else None))

    def widget(label, default):
        return widget_values.get(label, default)

    st.set_page_config = lambda **k: None
    st.markdown = lambda *a, **k: rec("markdown", *a)
    st.title = st.header = st.subheader = st.caption = st.write = lambda *a, **k: rec("text", *a)
    st.error = lambda *a, **k: st.errors.append(a[0])
    st.info = st.warning = st.success = lambda *a, **k: rec("note", *a)

    def stop():
        raise StopApp()
    st.stop = stop

    def metric(label, value, delta=None, **k):
        assert value is not None and "nan" not in str(value).lower(), f"metric {label} has invalid value {value}"
        rec("metric", label)
    st.metric = metric

    def dataframe(df, **k):
        assert isinstance(df, pd.DataFrame) and len(df) >= 0
        rec("dataframe", df.shape)
    st.dataframe = dataframe
    st.table = dataframe

    def pyplot(fig, **k):
        assert fig.get_axes(), "empty figure"
        assert any(ax.get_title(loc=l) for ax in fig.get_axes() for l in ('left', 'center', 'right')) or fig._suptitle is not None, "chart without title"
        assert any(ax.get_xlabel() or ax.get_ylabel() for ax in fig.get_axes()), "chart without axis labels"
        rec("pyplot", fig)
    st.pyplot = pyplot

    def image(path, **k):
        assert Path(path).exists(), f"missing image {path}"
        rec("image", path)
    st.image = image
    st.download_button = lambda *a, **k: rec("download", *a)

    st.radio = lambda label, options, **k: widget(label, options[0])
    st.selectbox = lambda label, options, **k: widget(label, list(options)[0])
    st.multiselect = lambda label, options, default=None, **k: widget(label, default or [])
    st.slider = lambda label, lo, hi, value=None, step=None, **k: widget(label, value if value is not None else lo)
    st.number_input = lambda label, lo=None, hi=None, value=None, step=None, **k: widget(label, value)

    def columns(spec):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Ctx(st) for _ in range(n)]
    st.columns = columns
    st.tabs = lambda names: [_Ctx(st) for _ in names]
    st.sidebar = _Ctx(st)
    st.cache_data = lambda f=None, **k: f if f else (lambda g: g)
    return st
