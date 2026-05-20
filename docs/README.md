# SOMA Docs

The SOMA documentation is open source and thrives on community contributions. Whether you’re fixing a typo, clarifying explanations, or adding entirely new content, your work benefits the whole community. This page explains how to contribute to the documentation using either GitHub’s web editor or your local development environment.

## Follow the style guide

All documentation changes must follow the SOMA style guide. Reviewers will provide feedback to ensure consistency in tone and quality. Don’t be discouraged if your pull request (PR) receives multiple review comments, as this process helps maintain clarity and uniformity across all docs. After your PR is merged, future updates may refine your content further.

When writing, keep these key principles in mind:

- Use active voice.
- Write in present tense.
- Be clear and concise. Use only as many words as needed.

## GitHub web editor

If you’re new to Git or prefer a simpler workflow, you can make small edits directly in GitHub’s web interface.

- Add a new page
  1. Go to the `src/content/docs` directory.
  2. Open the relevant subdirectory.
  3. Click Add file → Create new file.
  4. Write your content and commit your changes.

- Edit an existing page
  1. From the documentation website, you can use the "Edit this page" link at the bottom of each documentation page.
  2. From GitHub, navigate to the file you want to update. Click the pencil icon in the top-right.
  3. Make your edits and commit them.

## Set up a local environment

Cloning the documentation locally is recommended when you are creating larger, more significant changes to the docs. Fork and clone the [SOMA docs repository](~https://github.com/soma-org/docs~) locally. Documentation is located in the `src/content/docs` directory.

1. Install dependencies
   - If you use [Visual Studio Code](~https://code.visualstudio.com/~), install the [Prettier extension](~https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode~) to keep formatting consistent.
2. Make your changes
   - Edit or add files in the `src/content/docs` directory.
   - Stage and commit changes:
     ```bash
     git add .
     git commit -m "Describe your changes"
     git push
     ```
3. Preview locally
   - Install dependencies (if you don’t have `pnpm` installed, see the [pnpm installation guide](~https://pnpm.io/installation~)):
     ```bash
     pnpm install
     ```
   - Start the local dev server:
     ```bash
     pnpm run dev
     ```
   - Open `http://localhost:4321` to verify your updates.

## Review process

When your changes are ready:

1. Submit a PR to the `main` branch of the [SOMA docs repository](~https://github.com/soma-org/docs~).
2. A preview deployment will be generated so you can verify your changes. The preview is what you can expect to see online after your changes have been merged.
3. Reviewers will provide feedback. It’s your responsibility to update your PR based on their comments. Multiple reviewers might give input.
4. After at least one reviewer approves your PR, it gets merged into `main`, and your contribution goes live. Changes are reflected on the live website within 5-10 minutes after the PR has merged into `main`.

## Style Guide

### Editorial considerations

- Use simple words and concise sentences.  
  Prefer plain, direct language over complex or academic phrasing. Short sentences improve readability and are easier to localize.

- Avoid redefining common words.  
  Do not give familiar words new or unexpected meanings (for example, do not use "object" to mean something other than its standard technical or everyday sense). This prevents confusion, especially for new readers.

- Use technical terms with care.  
  Introduce technical terms only when necessary. Define them clearly the first time, and use them consistently throughout the documentation.

- Avoid jargon and slang.  
  Do not assume readers understand informal expressions, company-specific shorthand, or unnecessary buzzwords. Use precise terms instead.

- Prefer active, descriptive phrasing.  
  Instead of vague phrases like, "do the thing," explain the action explicitly: "Deploy the contract" or "Restart the node."

- Write for a global audience.  
  Keep in mind that many readers are non-native English speakers. Favor clarity over cleverness, and avoid idioms or culturally specific references.

---

### Spelling and grammar

#### Spelling

Use US English spelling in source content.

#### Avoid Latin abbreviations

Because many languages are not Latin-based, avoid using Latin abbreviations (e.g., i.e., etc., et. al, and so on). Prefer ex. or complete phrases like "for example", "and so on", and similar.

#### Grammar

##### Active voice

Use active voice whenever possible. Active voice is direct, clear, and uses fewer words. Passive voice is often less clear, awkward, and uses more words.

> ✅ Active: She installed the software.

> ❌ Passive: The software was installed by her.

##### Person

- First person → I or we

- Second person → you

- Third person → he, she, they, it, product names

Use second person ("you"). Do not use first or third person ("I" or "we").

> ✅ You can view the transaction history in the SOMA Explorer.

> ❌ We can view the transaction history in the SOMA Explorer.

##### Present tense

Use present tense whenever possible. Reserve future tense only for events that will happen at a specific future time, such as a scheduled product release.

Do not use future tense when describing product behavior or writing task instructions. From the reader's perspective, actions occur in the present as they follow the steps.

> Example: Present tense
>
> Click Save to save the updated file.  
> When you click Save, your device writes the changes to disk.  
> To save a file after you modify it, click Save.

> Example: Future tense (avoid)
>
> Your changes will be saved when you click Save.  
> When you click Save, the file will be written to disk.

Although technically correct, the future tense creates distance between the user and the action. It also makes the text harder to understand for ESL readers and more difficult to localize. In reality, the action happens immediately when the user clicks Save.

##### Punctuation

1. Sentences
   - Use a period at the end of a complete sentence.

   - Use a single space after a period (never two).

2. Lists
   - Full sentences in lists: End each item with a period.

   - Fragments or single words in lists: Do not use periods.

   - Mixed lists: Avoid mixing fragments and full sentences. Rewrite for consistency.

3. Parentheses
   - If the entire sentence is inside parentheses, place the period inside.

   - If the parentheses are part of a sentence, place the period outside.

   - Never place a period before the closing parenthesis.

4. Abbreviations
   - Keep the period as part of an abbreviation. However, [do not use Latin abbreviations](~#avoid-latin-abbreviations~).

5. Headings and titles
   - Do not use periods after headings, subheadings, or titles (unless the title ends in an abbreviation).

6. Numbers and decimals
   - Do not add an extra period after a decimal number.

##### Parentheses

Use parentheses to add clarifying or supplemental information that is not essential to the main sentence.

Avoid overusing parentheses. If the information is important, integrate it into the sentence instead of isolating it.

#### Avoid using (s) for plurals

Use “(s)” only if required for legal, contractual, or regulatory text where precision demands explicit acknowledgment of both singular and plural forms. Otherwise, use the plural form without parentheses.

##### Oxford (serial) commas

Do use serial commas.

> ✅ You must install Cargo, Rust, Docker, and the SOMA CLI to create a SOMA node.

> ❌ You must install Cargo, Rust, Docker and the SOMA CLI to create a SOMA node.

##### Numbers

Do not write out numbers when referring to a number of items; always use the numerical value.

> The folder contains 24 files.
> One folder contains 7 files, and the other contains 24 files.
> At least 20 pieces of candy fell off the table.

Do write out numbers when they are grammatically part of the sentence.

> One can always include extra documentation to support the theory.
> The client checks if the checkpoint is the last one of the epoch.

##### Quotation marks

Do not use quotation marks, except for the single exception "Hello, World!".

##### Ampersands

Do not use ampersands (`&`) in content to replace `and` as the word is more accessible and less error-prone for some programmatic use cases. If you absolutely must include an ampersand in content, escape it using `&amp;`.

##### Exclamations

Do not use exclamation marks. If you'd like to express excitement, such as:

> Congratulations! You've finished the tutorial.

Use the confetti emoji instead:

> Congratulations, you've finished the tutorial. 🎉

---

### Terminology and vocabulary

Maintain a consistent vocabulary throughout the documentation. Define project-specific terms clearly on first use, and use them consistently. Capitalize product names and proper nouns. Use lowercase for general concepts.

#### Proper nouns

Capitalize proper nouns throughout.

Proper nouns include:

- Names of people: Bob Ross

- Named places: San Francisco, Union Station

- Products and services: Slack, Google Play

- Trademarks: Coca-Cola

- Book titles: The Move Book

- Standards or technologies: Local Area Network (LAN)

#### Product names

Product names are proper nouns. Capitalize all words of a product name. When referring to a product, use only the product name without "the".

#### Acronyms and abbreviations

##### Acronyms

Spell out a term or phrase on first use in a topic, followed by the acronym in parentheses. Then use the acronym for subsequent mentions.

> Example: You can mint non-fungible tokens (NFTs) using your SOMA Wallet. To view an NFT after you mint it, click the NFTs tab of your wallet.

##### Terms that should always be used as acronyms

- CLI
- SDK

##### Abbreviations

Abbreviations for words should not be used. Write out the full word for clarity.

> ✅ Open the tab for more information.

> ❌ Open the tab for more info.

---

### Capitalization

#### Title capitalization

For title capitalization, follow these guidelines:

- Do not capitalize short conjunctions and prepositions (a, an, and, but, for, in, or, so, to, with, yet), unless they are the first or last word.

- Capitalize all other words (including 'Is' and 'Be' as they are verbs).

- Capitalize the word after a hyphen.

- Match casing for commands or special terms, such as cURL.

- Match the casing for API elements and programming language keywords.

#### Section heading capitalization

Use sentence capitalization for section headings, table cells or headers, list items, captions, alt text, and error messages.

#### Body text capitalization

##### Do:

Always capitalize the first word of a new sentence.

Always capitalize proper nouns and product names. See [words to always capitalize](~#word-list~) for the exhaustive list of capitalized terms.

##### Do not:

Do not use all uppercase for emphasis; use bold instead.

> Example: IMPORTANT vs Important

Do not use bicapitalization or internal capitalization unless it is part of a brand.

> Examples: YouTube or DreamWorks.

Do not capitalize the spelled-out form of an acronym unless it's a proper noun.

> Example: HyperText Markup Language (HTML).

---

### Body text styling

#### Bold text

Use bold for UI elements that appear on the screen, such as buttons, menu items, field labels, and commands.

> Example: Click Save to store your changes.

Use bold sparingly for emphasis and only when necessary for clarity. Avoid overusing bold for general emphasis in body text.

#### Keyboard button text

Use the `<kbd>` tags around text that corresponds to a physical button on the keyboard.

Example:

Press <kbd>0</kbd>, <kbd>1</kbd>, or <kbd>2</kbd> to select a key scheme and then press <kbd>Enter</kbd>.

Markdown:

```
Press <kbd>0</kbd>, <kbd>1</kbd>, or <kbd>2</kbd> to select a key scheme and then press <kbd>Enter</kbd>.
```

#### Italic text

Use italics when introducing a new term for the first time.

> Example: The term for the cost of processing transaction blocks is _*gas*_.

#### Slashes

Do not use slashes in place of "and" or "or", such as True / False or True/False. Use True or False, or True | False in code documentation.

---

### Titles and headings

Use enough words in headings and titles to make it easy to know which link to click on a search results page. One-word titles (for example, Installing) do not provide enough information to determine the contents of a topic.

#### Page titles

Use descriptive titles that include relevant keywords so readers can quickly identify the content. Shorter titles are preferred in the navigation pane. You can set a different navigation title using the `sidebar.label` property in the document frontmatter.

Readers usually search for information to complete a specific task. Avoid vague titles such as Get Started. Get started with what? If there are multiple products or features, the meaning is unclear.

A better option is Get Started with SOMA, but even this is still too broad. Readers want guidance for a specific task or journey. Instead, use precise titles such as Create a SOMA Full Node or Install SOMA Tooling. These tell the reader exactly what they will learn or accomplish.

#### Page headings

Use heading capitalization style (sentence case). Do not stack headings (place two one after the other without body text in between them).

If something is formatted as inline code in the body, format it the same in the heading.

Do not use a page title as a heading in a different page. This can interfere with search result accuracy. Page titles should be unique and descriptive, while headings can be reused.

> ✅ Correct usage:
> Page 1: Page title "SOMA Gas Profiling" with heading "Environment configuration"
> Page 2: Page title "SOMA Indexing" with heading "Environment configuration"

> ❌ Incorrect usage:
> Page 1: Page title "SOMA Gas Profiling" with heading "Environment configuration"
> Page 2: Page title: "SOMA Features" with heading "SOMA gas profiling"

#### Heading sizes

- Heading 1: (#) Reserved exclusively for the page title. When the title is specified in the frontmatter, the page is formatted automatically with that title as a Heading 1 element.

- Heading 2: (##) Top-level section headings. Used to introduce new topics on a page.

- Heading 3: (###) Sub-topics for each Heading 2. Used to introduce multiple concepts under a top-level heading.

- Heading 4: (####) Sub-topics for each Heading 3. Used to introduce examples for Heading 3 topics or format sub-sections distinctly.

- Heading 5: (#####) Sub-topics for each Heading 4. When formatted, looks identical to bolded body text, but slightly larger. Can also be used as an alternative to bolded text to create unique formatting within other elements, such as:

Example:

> ##### Heading 5 text.

> **Bold body text.**

> Normal body text.

Markdown:

```
> ##### Heading 5 text.

> Bold body text.

> Normal body text.
```

#### Code elements in section headings

If a word or phrase is formatted in the page content as `inline code`, then it should be formatted the same in a section heading.

> ##### Section heading: Install `soma-cli`

> Body content: Install `soma-cli` using the command: ...

See [inline code](~#inline-code~) for more information.

---

### Lists

Use lists to present items or steps clearly. Introduce a list with a short description ending in a colon (:). Lists should be used in place of sentences that include more than 4 items as a serial comma list.

All lists should use sentence capitalization unless listing the titles of documentation pages, in which case the title case should be respected.

> Title case example: The Build section includes:
>
> - Building with SOMA
> - Using the CLI to Start a Network
> - Creating Smart Contracts
> - SOMA Tutorial
> - SOMA Examples

> Sentence case example: The Build section includes:
>
> - For objects, the `tx.object(objectId)` function is used to construct an input that contains an object reference.
> - For pure values, the `tx.pure(type, value)` function is used to construct an input for a non-object input.

#### Numbered lists

Use when items must be done in order, describe a sequence, or describe a specific number of items.

Example:

1. Create a fork of the repo.
2. Clone your fork of the repo.
3. Install SOMA.

Markdown:

```
1. Create a fork of the repo.
1. Clone your fork of the repo.
1. Install SOMA.
```

#### Bulleted lists

Use for related items that do not need order. Use sentence capitalization and consistent punctuation. Add periods only if the item is a full sentence.

Example:

The documentation site supports the following browsers:

- Firefox version X or later
- Chrome version X or later
- Edge version X or later

Markdown:

```
The documentation site supports the following browsers:

- Firefox version X or later
- Chrome version X or later
- Edge version X or later
```

#### Term lists

Use to define terms or concepts. The term should be bold text, followed by a colon (:) and the term's definition using sentence capitalization:

> - **Term**: Sentence capitalization used for the term definition.

Example:

- **Term**: A description of the term.

- **DAG**: A directed acyclic graph (DAG) is a data modeling or structuring tool typically used in data architectures.

Markdown:

```
- **Term**: A description of the term.

- **DAG**: A directed acyclic graph (DAG) is a data modeling or structuring tool typically used in data architectures.
```

#### Related links list

At the bottom of a page, you can direct the reader to additional, related content via a related links list. Use the `RelatedLink` component:

External links:

```jsx
<RelatedLink
  href="/path/to/page"
  label="Page Title"
  desc="Description of page."
/>
```

Internal links:

```jsx
<RelatedLink to="/guides/somepage.mdx" />
```

#### Attribute lists

Use lists with inline code formatting to list attributes for components such as objects.

Example:

An event object in SOMA consists of the following attributes:

- `id`: JSON object containing the transaction digest ID and event sequence.
- `packageId`: The object ID of the package that emits the event.
- `transactionModule`: The module that performs the transaction.
- `sender`: The SOMA network address that triggered the event.
- `type`: The type of event being emitted.
- `parsedJson`: JSON object describing the event.
- `bcs`: Binary canonical serialization value.
- `timestampMs`: Unix epoch timestamp in milliseconds.

Markdown:

```
An event object in SOMA consists of the following attributes:

- `id`: JSON object containing the transaction digest ID and event sequence.
- `packageId`: The object ID of the package that emits the event.
- `transactionModule`: The module that performs the transaction.
- `sender`: The SOMA network address that triggered the event.
- `type`: The type of event being emitted.
- `parsedJson`: JSON object describing the event.
- `bcs`: Binary canonical serialization value.
- `timestampMs`: Unix epoch timestamp in milliseconds.
```

---

### Tables

#### Table headings

Capitalize the first word in the heading. Center align the text. Bold labels in the Header row.

Example:

| Column one  | Column two | Column three | Column four  |
| :---------- | :--------: | :----------: | :----------- |
| Metric name |     10     |      X       | Text string. |

Markdown:

```

| Column one | Column two | Column three | Column four |

```

#### Table alignment

Center align labels in the Heading row. Left align strings of text. Center align values and Xs or checkmarks.

Example:

| Column one  | Column two | Column three | Column four  |
| :---------- | :--------: | :----------: | :----------- |
| Metric name |     10     |      X       | Text string. |

Markdown:

```

| Column one | Column two | Column three | Column four |
| :--- | :---: | :---: | :--- |

```

#### Table text

Follow style guidelines for regular body text.

---

### Code

Words or phrases that refer to functions, object names, CLI tool names, or CLI commands should be formatted as inline code when used in a sentence.

Use codeblocks for larger sections. Always use actual codeblocks (no images) formatted with the correct syntax highlighting.

#### Inline code

Use backticks (\`) around inline code.

Things that should always be formatted as inline code (body and headings) include:

- Object names: `myObject`

- Function names: `HelloWorld`

- File names with extensions: `myPackage.move`

- File extensions: `.jpg`

- CLI tool names: `brew`

- CLI commands when used within a sentence: If using the suggested location, type `export PATH=$PATH:~/bin` and press Enter.

- Variable names: `PATH`

- File paths: `~/.cargo/bin`

#### Console commands

Console commands must be formatted using a fenced code block with the `bash` (or `sh`) language identifier. Starlight uses [Expressive Code](~https://expressive-code.com/~) and automatically renders shell language blocks in a terminal window frame. Do not prefix commands with `$`.

Example:

```bash
brew install node
```

Markdown:

````
```bash
brew install node
```
````

You can add a title to describe the terminal action:

````
```bash title="Installing dependencies"
pnpm install
```
````

Keep command and response outputs in different code blocks. This ensures commands can be copied and run correctly.

#### Codeblocks

Introduce codeblocks with descriptive text, including where the code should be placed within a project:

> Example: Create a new file in the `src` directory with the name `config.ts` and populate the file with the following code:

Follow with explanations:

> Example: There are a few details to note in this code:
>
> 1. The first line imports the required configuration module.
> 2. The `Config` interface defines the shape of the configuration object.
> 3. Two error codes ensure the program handles invalid states correctly.

Use three backticks (\`\`\`) to initiate, followed by the code's language (TypeScript, Python, etc.) for proper syntax highlighting, and a title indicating the file name. Starlight uses [Expressive Code](~https://expressive-code.com/~) for code block rendering. You can highlight specific lines with `{1,3-5}` markers or mark inserted/deleted lines with `ins={1}` and `del={2}` after the language identifier.

Example:

```ts title='config.ts'
import { defineConfig } from "./utils";

interface Config {
  port: number;
  debug: boolean;
}

// Error codes
const ERR_INVALID_PORT = 1;
const ERR_MISSING_CONFIG = 2;
```

Markdown:

````
```ts title='config.ts'
import { defineConfig } from './utils';

interface Config {
  port: number;
  debug: boolean;
}

// Error codes
const ERR_INVALID_PORT = 1;
const ERR_MISSING_CONFIG = 2;
```
````

---

### Procedures, tasks, and instructions

Introduce a procedure with an infinitive verb. Format procedures using a numbered or ordered list.

#### Keyboard keys in procedures

When you provide instructions to press keyboard keys, such as Press Enter to continue, use uppercase for the key name and format the key name as bold text.

Example:

To get the latest version of the extension:

1. Open Google Chrome.
2. Click **Extensions**, then click **Manage Extensions**.
3. Click **Details** for the extension, then click **View in Chrome Web Store**.

Markdown:

```

To get the latest version of the extension:

1. Open Google Chrome.
1. Click **Extensions**, then click **Manage Extensions**.
1. Click **Details** for the extension, then click **View in Chrome Web Store**.

```

#### UI elements

Format UI elements, such as field labels, button names, and menu commands, in bold text. Always match the exact text or label of the UI element, including capitalization. Do not include special characters, such as ellipses, if included in the element label.

> Example: Click **More Transactions** to open the **Transactions** page.

---

### Links and references

Always use full, relative links when linking to topics on the documentation site.

For the link text, use either:

- The topic title of the target topic, respecting the title case format.

- A portion of the sentence that serves as the link text for the link in a list or "Learn more" sentences. Do not use a URL as the link text.

Example:

> To learn more, see the API Reference.

Markdown:

```markdown
To learn more, see the [API Reference](~/reference/api.mdx~).
```

Use keywords from the target topic title when using inline links.

Example:

> Before you get started, make sure to install the prerequisites.

Markdown:

```

Before you get started, make sure to install the [prerequisites](/guides/getting-started.mdx#prerequisites).

```

#### URLs and web addresses

Create a link with descriptive text to a site or URL. Provide the URL only when a reader needs to copy it, such as in example code or configuration files.

#### Referring to pages in our docs

Refer to pages in the documentation set as "topic"s. A "guide" can comprise many related topics.

> Example: See the Install topic in the Validator guide for more information.

---

### Starlight components

This project uses [Starlight](~https://starlight.astro.build~) and all documentation pages are `.mdx` files. Starlight provides built-in components for common documentation patterns. Import them from `@astrojs/starlight/components` at the top of your MDX file, below the frontmatter.

```mdx
---
title: Example Page
---

import {
  Aside,
  Card,
  CardGrid,
  Steps,
  Tabs,
  TabItem,
  Badge,
  LinkCard,
} from "@astrojs/starlight/components";
```

#### Asides

Use asides (also known as callouts) to highlight supplementary information. Starlight supports a Markdown shorthand syntax using `:::` fences as well as the `<Aside>` component. Prefer the Markdown shorthand for simple callouts:

```mdx
:::note
Include nonessential, supplementary information here.
:::

:::tip
Helpful advice for the reader.
:::

:::caution
Warn the reader about a potential pitfall.
:::

:::danger
Critical warning about a destructive or irreversible action.
:::
```

Use the `<Aside>` component when you need a custom title or icon:

```mdx
<Aside type="caution" title="Watch out!">
  A warning aside with a custom title.
</Aside>
```

#### Cards and link cards

Use `<Card>` and `<CardGrid>` to group related content visually. Use `<LinkCard>` to prominently link to other pages:

```mdx
<CardGrid>
  <Card title="First concept" icon="star">
    Description of the first concept.
  </Card>
  <Card title="Second concept" icon="rocket">
    Description of the second concept.
  </Card>
</CardGrid>

<LinkCard
  title="Learn more"
  href="/concepts/targets"
  description="Read about SOMA targets."
/>
```

#### Steps

Use `<Steps>` to wrap ordered lists in step-by-step guides for clearer visual hierarchy:

```mdx
<Steps>
  1. Install the CLI. 2. Initialize your project. 3. Run the dev server.
</Steps>
```

#### Tabs

Use `<Tabs>` and `<TabItem>` to group equivalent content where a reader only needs one option (for example, commands for different package managers or operating systems):

````mdx
<Tabs>
  <TabItem label="npm">```bash npm install ```</TabItem>
  <TabItem label="pnpm">```bash pnpm install ```</TabItem>
</Tabs>
````

Use the `syncKey` attribute to keep multiple tab groups on the same page synchronized.

#### Badges

Use `<Badge>` to display small status or category labels inline:

```mdx
<Badge text="New" variant="tip" />
<Badge text="Deprecated" variant="caution" />
```

For the full list of available components and their props, see the [Starlight Components documentation](~https://starlight.astro.build/components/using-components/~).

---

### Images

#### Image storage

Store images in `src/assets/` so that Astro can optimize and transform them at build time. Do not store images in the `public/` directory unless they must remain unprocessed (for example, favicons or Open Graph images).

Use a relative path from the content file to reference the image:

```mdx
![A diagram showing the network topology](../../assets/images/network-diagram.png)
```

For more control over image attributes (dimensions, quality, format), import the image and use Astro's `<Image />` component:

```mdx
import { Image } from "astro:assets";
import networkDiagram from "../../assets/images/network-diagram.png";

<Image src={networkDiagram} alt="A diagram showing the network topology" />
```

#### Image format

Use `.png` for screenshots, diagrams, and images with text. Use `.svg` for icons and simple illustrations. Avoid `.jpg` unless the image is a photograph.

Astro automatically converts optimized images to `.webp` at build time when they are stored in `src/assets/`, so you do not need to manually convert formats.

#### Image resolution

Images should be at least 400 pixels wide. If an image looks blurry when uploaded, try making a new image in higher resolution.

#### Alt text and captions

Use alt text to describe what the image shows. Use the caption to explain why the image is meaningful in the context of the page. See [Accessibility](~#accessibility~) considerations for more guidance.

---

### Index pages

Section index pages must link to subcategory index pages when one exists. This is to ensure users can easily navigate deeper into subsections without relying solely on the sidebar and creates a consistent navigation structure.

For example, if the docs folder looks like this:

```
/docs/
  guides/
    index.mdx
    setup/
      index.mdx
      install.mdx
```

Then `/docs/guides/index.mdx` must include a link to `/docs/guides/setup/index.mdx` rather than a link to `/docs/guides/setup/install.mdx`.

---

### Accessibility

Reference works for making content accessible:

- [A11Y Style Guide](~https://a11y-style-guide.com/style-guide/~)
- [Bitsofcode Accessibility Cheatsheet](~https://bitsofco.de/the-accessibility-cheatsheet/~)
- [Atlassian Design System - Inclusive writing reference](~https://atlassian.design/content/inclusive-writing~)
- [MailChimp's writing style guide](~https://styleguide.mailchimp.com/writing-for-accessibility/~)
- [Microsoft Style Guide Accessibility Terms](~https://learn.microsoft.com/en-us/style-guide/a-z-word-list-term-collections/term-collections/accessibility-terms~)
- [Writing for All Abilities](~https://learn.microsoft.com/en-us/style-guide/accessibility/writing-all-abilities~) (Microsoft Style Guide)

#### Formatting

Do not use color or special symbols to add emphasis to text. Screen readers are designed to interpret bold (`<strong>`) and italic (`<em>`) in web pages.

#### Images

Add captions and alt text that describe the image for someone using a screen reader. What are the important details in the image that someone using a screen reader can't see?

Use alt text to describe what the image shows. Use the caption to explain why the image is meaningful in the context of the page.

An image is not a substitute for text; images should only supplement text. Do not rely on an image to convey information not in text form. For example, an image of a table of values does no one any good if the image fails to display for a host of possible reasons.

---

### Reference style guides

- [Write the Docs Style Guide article](~https://www.writethedocs.org/guide/writing/style-guides/~)
- [GitLab Style Guide](~https://docs.gitlab.com/ee/development/documentation/styleguide/index.html~) - managed as a community project
- [Digital Ocean Style Guide](~https://www.digitalocean.com/community/tutorials/digitalocean-s-technical-writing-guidelines~)
- [SUSE Style Guide](~https://documentation.suse.com/style/current/#sec-techwriting~)
- [Microsoft Style Guide](~https://docs.microsoft.com/en-us/style-guide/welcome/~)
- [Google Developer Style Guide](~https://developers.google.com/style~)
- [CDN Language and Style Reference](~http://cdn.static-economist.com/sites/default/files/pdfs/style_guide_12.pdf~)
