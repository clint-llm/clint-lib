use serde::Serialize;
use thiserror;
use tinytemplate;
use tinytemplate::{format_unescaped, TinyTemplate};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("template error: {0}")]
    TemplateError(#[from] tinytemplate::error::Error),
}

type Result<T> = core::result::Result<T, Error>;

pub fn render_template(template: &str, context: &impl Serialize) -> Result<String> {
    let mut tt = TinyTemplate::new();
    tt.set_default_formatter(&format_unescaped);
    tt.add_template("x", template)
        .map_err(Error::TemplateError)?;
    tt.render("x", &context).map_err(Error::TemplateError)
}
