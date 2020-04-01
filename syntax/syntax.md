# 语法

本部分对常用的NEON指令进行介绍，会先介绍NEON intrinsics函数 ，然后再介绍与之对应的NEON汇编指令，最后跟着简单的例子。（实际开发中，即便是写汇编代码，使用intrinsics也有好处。先用intrinsics写好代码编译后在反汇编，在此基础上进行优化，可能比较省力。）

### 加载元素 {#load_data}

To create a heading, add one to six `#` symbols before your heading text. The number of # you use will determine the size of the heading.

opening curly bracket (separated from the text with a least one space), a hash, the ID and a closing curly bracket, the ID is set on the header. If you use the trailing hash feature of atx style headers, the header ID has to go after the trailing hashes. For example:

```markdown
Hello {#id}
-----

# Hello {#id}

# Hello # {#id}
```

### 存储元素 {#store_data}

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