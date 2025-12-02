# -----------------------
# Present detected client and models (REPLACEMENT)
# -----------------------
st.sidebar.markdown("### GenAI Client / Model selector (debug)")
st.sidebar.write(f"Detected client kind: **{client_label}**")
st.sidebar.write(f"GEN_CLIENT initialized: **{bool(GEN_CLIENT)}**")
st.sidebar.write(f"GEMINI_KEY present: **{bool(GEMINI_KEY)}**")

# If no models discovered, populate safe fallbacks depending on client kind
if not available_models:
    if GEN_CLIENT_KIND == "new":
        available_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    elif GEN_CLIENT_KIND == "legacy":
        available_models = ["text-bison-001", "chat-bison-001"]
    else:
        # Unknown client; provide both kinds so user can try either provider
        available_models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "text-bison-001",
            "chat-bison-001",
        ]

# Show a few models in the sidebar for user visibility
st.sidebar.markdown("Available model examples (first few):")
for m in available_models[:10]:
    st.sidebar.text(m)

# Allow manual model entry and explicit 'use manual' toggle so selectbox never becomes empty
manual_toggle = st.sidebar.checkbox("Use manual model name instead of dropdown", value=False)
manual_model = None
if manual_toggle:
    manual_model = st.sidebar.text_input("Manual model name (paste exact model id)", value="")

# Build dropdown options (guaranteed non-empty now)
model_dropdown_options = available_models.copy()
# Add an explicit 'auto' option at top for automatic choice logic
if "auto" not in model_dropdown_options:
    model_dropdown_options.insert(0, "auto")

model_choice = st.sidebar.selectbox("Model to use (dropdown)", options=model_dropdown_options, index=0)

# If user chose manual via toggle, prefer manual input
if manual_toggle and manual_model and manual_model.strip():
    chosen_model = manual_model.strip()
else:
    chosen_model = model_choice

# Display chosen model in sidebar for clarity
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Resolved model to use:** `{chosen_model}`")
st.sidebar.info(
    "If model calls fail with NotFound errors, try toggling 'Use manual model name' and paste a known model like "
    "`gemini-2.5-flash` (for google-genai) or `text-bison-001` (for legacy)."
)
