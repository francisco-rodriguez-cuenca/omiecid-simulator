"""Sphinx configuration of the OmieCID-Simulator documentation site.

The source folder is the repository root and this file is the configuration:

    python -m sphinx -T -W --keep-going -b html -c docs . <output>/html

so the links between README.md, the guides, the tutorials and CHANGELOG.md work unchanged on GitHub
and on the site. Only the files of ``include_patterns`` become pages. Notebooks are shown with their
saved outputs and are never re-run, so the build needs no JAX.
"""
import os
import tomllib
from pathlib import Path

from myst_parser.mdit_to_docutils.sphinx_ import SphinxRenderer

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "pyproject.toml", "rb") as fh:
    _project = tomllib.load(fh)["project"]

# ------------------------------------------------------------------ project
project = "OmieCID-Simulator"
author = "Francisco Rodriguez-Cuenca, Eugenio Francisco Sánchez Úbeda, Álvaro Jesús López López"
copyright = "2026, Francisco Rodriguez-Cuenca"
release = _project["version"]
version = release

# ------------------------------------------------------------------ sources
root_doc = os.path.join("docs", "index")
include_patterns = [
    "README.md",
    "CHANGELOG.md",
    "docs/index.md",
    "docs/evidence.md",
    "docs/citing.md",
    "docs/api.md",
    "docs/market-model.md",
    "docs/modelling-assumptions.md",
    "docs/sessions.md",
    "docs/data.md",
    "docs/validation.md",
    "docs/testing.md",
    "docs/performance.md",
    "docs/limitations.md",
    "tutorials/README.md",
    "tutorials/T0[1-9]_*.ipynb",
    # the API reference pages that sphinx-autoapi writes (see "API reference" below)
    "autoapi/**",
]
exclude_patterns = ["_build"]

# napoleon reads the Google-style docstrings ("Args:", "Returns:", "Attributes:") of the package;
# like sphinx-autoapi it works on the source text and imports nothing.
extensions = ["myst_nb", "autoapi.extension", "sphinx.ext.napoleon"]
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 4

# Notebooks are shown with the outputs saved in them, never executed.
nb_execution_mode = "off"
# A re-executed notebook splits its stdout into several stream outputs; show them as one block.
nb_merge_streams = True

# ------------------------------------------------------------------ API reference
# sphinx-autoapi reads the package source as text and never imports it, so the build needs no JAX.
# The reference covers a curated list of public modules: the session (Session, AgentOrder,
# FillReport, TopOfBook), the contract book, contracts, order types, settlement, encoding, the
# market configuration, and the data loader, data configuration and tape synthesizer. Left out: the
# inherited order-book core jaxob/, private (underscore) modules, and the internal modules below.
autoapi_type = "python"
autoapi_dirs = [str(ROOT / "omiecid_simulator")]
autoapi_root = "autoapi"
# Module names start at the folder above the package ("omiecid_simulator.omiecid...") even when that
# folder holds an __init__.py of its own, as the research tree does.
autoapi_python_use_implicit_namespaces = True
autoapi_ignore = [
    "*/omiecid_simulator/jaxob/*",
    "*/_[!_]*.py",
    "*/_[!_]*/*",
    "*/omiecid_simulator/omiecid/iceberg_manager.py",
    "*/omiecid_simulator/omiecid/jax_constants.py",
    "*/omiecid_simulator/omiecid/omie_states.py",
    "*/omiecid_simulator/omiecid/order_validator.py",
    "*/omiecid_simulator/omiecid/trade_extraction.py",
    "*/omiecid_simulator/jaxomiecid/omie_cache.py",
]
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
# docs/api.md holds the reference's table of contents.
autoapi_add_toctree_entry = False
autoapi_keep_files = False
# "Attributes:" sections as a field list: autoapi already lists each attribute once.
napoleon_use_ivar = True
# Names the README imports that their module's __all__ leaves out: shown all the same.
SHOW_DESPITE_ALL = {"omiecid_simulator.omiecid.omie_session.AGENT_TRADER_ID"}


def show_readme_names(app, what, name, obj, skip, options):
    """Show a ``SHOW_DESPITE_ALL`` name; leave every other decision to sphinx-autoapi."""
    return False if name in SHOW_DESPITE_ALL else None


# Two docstrings of the frozen package are not valid reStructuredText (an indented continuation line
# inside a block). They are shown as written, in a literal block after their first line, not parsed.
# Keys are (object type, name as sphinx-autoapi passes it: dotted for a module, bare for a function).
VERBATIM_DOCSTRINGS = {
    ("module", "omiecid_simulator.jaxomiecid.omie_data_loader"),
    ("function", "compute_agent_pnl"),  # omiecid_simulator.omiecid.settlement
}


def show_docstring_verbatim(app, what, name, obj, options, lines):
    """Render the docstring of a ``VERBATIM_DOCSTRINGS`` object unchanged, as a literal block."""
    if (what, name) in VERBATIM_DOCSTRINGS and len(lines) > 1:
        body = lines[1:]
        while body and not body[0].strip():
            body.pop(0)
        lines[1:] = ["", "::", ""] + [f"    {ln}" if ln.strip() else "" for ln in body]


suppress_warnings = [
    # a listed module imports from a module left out above (jaxob/, internal helpers): the import cannot be resolved
    "autoapi.python_import_resolution",
]

# ------------------------------------------------------------------ html
html_theme = "furo"
html_title = f"{project} {release}"

# ------------------------------------------------------------------ links to files that are not pages
# LICENSE, CITATION.cff, the constraints file, the evidence documents, the certification notebooks:
# such a link becomes a GitHub link at the built ref (the Read the Docs git identifier, else the tag).
GITHUB_BLOB = "https://github.com/francisco-rodriguez-cuenca/omiecid-simulator/blob/"
LINK_REF = os.environ.get("READTHEDOCS_GIT_IDENTIFIER") or f"v{release}"
_render_link_unknown = SphinxRenderer.render_link_unknown


def render_link_unknown(self, token):
    """A relative link to an existing file that is not a page of the site is rendered as a GitHub
    link. Any other link (a page, or a file that does not exist) gets MyST's normal handling, so a
    broken link still warns and fails the ``-W`` build."""
    env = self.sphinx_env
    href = self._handle_relative_docs(self.md.normalizeLinkText(token.attrGet("href") or ""))
    path, _, anchor = href.partition("#")
    if env.srcdir and path:
        _, abs_path = env.relfn2path(path, env.docname)
        if os.path.isfile(abs_path) and env.path2doc(abs_path) not in env.found_docs:
            rel = Path(os.path.normpath(abs_path)).relative_to(env.srcdir).as_posix()
            token.attrs["href"] = f"{GITHUB_BLOB}{LINK_REF}/{rel}" + (f"#{anchor}" if anchor else "")
            return self.render_link_url(token)
    return _render_link_unknown(self, token)


SphinxRenderer.render_link_unknown = render_link_unknown


# ------------------------------------------------------------------ the root page
def write_root_redirect(app, exception):
    """The home page is under docs/, so nothing writes the site's root page: write a redirect."""
    if exception is not None or app.builder.format != "html":
        return
    target = app.builder.get_target_uri(app.config.root_doc)
    Path(app.outdir, "index.html").write_text(
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\n"
        f"<meta http-equiv=\"refresh\" content=\"0; url={target}\">\n"
        f"<link rel=\"canonical\" href=\"{target}\">\n"
        f"<title>{project}</title></head>\n"
        f"<body><p><a href=\"{target}\">{project} documentation</a></p></body></html>\n",
        encoding="utf-8")


def setup(app):
    app.connect("build-finished", write_root_redirect)
    app.connect("autoapi-skip-member", show_readme_names)
    # before napoleon (priority 500), so the literal block is left as it is
    app.connect("autodoc-process-docstring", show_docstring_verbatim, priority=400)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
