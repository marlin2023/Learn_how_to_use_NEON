# Markdown

Most of the examples from this documentation are in Markdown. Markdown is default parser for GitBook, but one can also opt for the [AsciiDoc syntax](asciidoc.md).

Here’s an overview of Markdown syntax that you can use with GitBook (same as GitHub with some additions).

### Headings

To create a heading, add one to six `#` symbols before your heading text. The number of # you use will determine the size of the heading.

```markdown
# This is an <h1> tag
## This is an <h2> tag
###### This is an <h6> tag
```

GitBook supports a nice way for explicitly setting the header ID. If you follow the header text with an opening curly bracket (separated from the text with a least one space), a hash, the ID and a closing curly bracket, the ID is set on the header. If you use the trailing hash feature of atx style headers, the header ID has to go after the trailing hashes. For example:

```markdown
Hello {#id}
-----

# Hello {#id}

# Hello # {#id}
```

### Paragraphs and Line Breaks {#paragraphs}

A paragraph is simply one or more consecutive lines of text, separated by one or more blank lines. (A blank line is any line that looks like a blank line — a line containing nothing but spaces or tabs is considered blank.) Normal paragraphs should not be indented with spaces or tabs.

```
Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.
```

### Emphasis {#emphasis}

```markdown
*This text will be italic*
_This will also be italic_

**This text will be bold**
__This will also be bold__

~~This text will be crossed out.~~

_You **can** combine them_
```