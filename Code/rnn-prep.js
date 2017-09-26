"use strict";
const ts = require("typescript");
const fs = require("fs");
const path = require("path");
function print(x) { console.log(x); }
var removableLexicalKinds = [
    ts.SyntaxKind.EndOfFileToken,
    ts.SyntaxKind.NewLineTrivia,
    ts.SyntaxKind.WhitespaceTrivia
];
var templateKinds = [
    ts.SyntaxKind.TemplateHead,
    ts.SyntaxKind.TemplateMiddle,
    ts.SyntaxKind.TemplateSpan,
    ts.SyntaxKind.TemplateTail,
    ts.SyntaxKind.TemplateExpression,
    ts.SyntaxKind.TaggedTemplateExpression,
    ts.SyntaxKind.FirstTemplateToken,
    ts.SyntaxKind.LastTemplateToken,
    ts.SyntaxKind.TemplateMiddle
];
let root = "C:/Users/t-vihell/Documents/TypeScriptStudy/Clones";
let outputDir = "C:/Users/t-vihell/Documents/TypeScriptStudy/outputs-project/";
const ANY_THRESHOLD = 0.2;
fs.readdirSync(root).forEach(org => fs.readdirSync(root + "/" + org).forEach(project => traverseProject(org, project)));
function traverseProject(org, project) {
    let dir = root + "/" + org + "/" + project;
    let outFile = dir.substr(root.length + 1) + ".json";
    outFile = outFile.replace(/\//g, "__");
    outFile = outputDir + outFile;
    if (fs.existsSync(outFile))
        return;
    let projectTokens = traverse(dir);
    fs.writeFileSync(outFile, projectTokens.join("\n"), 'utf-8');
}
function traverse(dir) {
    var children = fs.readdirSync(dir);
    let tokens = [];
    if (children.find(value => value == "tsconfig.json")) {
        print("Config in: " + dir);
        tokens = tokens.concat(extractAlignedSequences(dir));
    }
    else {
        children.forEach(function (file) {
            let fullPath = dir + "/" + file;
            if (fs.statSync(fullPath).isDirectory()) {
                if (fullPath.indexOf("DefinitelyTyped") < 0) {
                    tokens = tokens.concat(traverse(fullPath));
                }
                else {
                    print("Skipping: " + fullPath);
                }
            }
        });
    }
    return tokens;
}
function extractAlignedSequences(inputDirectory) {
    const keywords = ["async", "await", "break", "continue", "class", "extends", "constructor", "super", "extends", "const", "let", "var", "debugger", "delete", "do", "while", "export", "import", "for", "each", "in", "of", "function", "return", "get", "set", "if", "else", "instanceof", "typeof", "null", "undefined", "switch", "case", "default", "this", "true", "false", "try", "catch", "finally", "void", "yield", "any", "boolean", "null", "never", "number", "string", "symbol", "undefined", "void", "as", "is", "enum", "type", "interface", "abstract", "implements", "static", "readonly", "private", "protected", "public", "declare", "module", "namespace", "require", "from", "of", "package"];
    let files = [];
    walkSync(inputDirectory, files, '.ts');
    let program = ts.createProgram(files, { target: ts.ScriptTarget.Latest, module: ts.ModuleKind.CommonJS });
    let checker = program.getTypeChecker();
    let allFileTokens = [];
    for (const sourceFile of program.getSourceFiles()) {
        let filename = sourceFile.getSourceFile().fileName;
        if (filename.endsWith('.d.ts'))
            continue;
        try {
            let relativePath = path.relative(inputDirectory, filename);
            if (relativePath.startsWith(".."))
                continue;
            let data = extractTokens(sourceFile, checker);
            if (data[0].length != data[1].length)
                console.log(data[0].length + ", " + data[1].length);
            // Remove distinct numerals, string, regexes from data, remove any internal white-space from tokens
            for (var ix in data[0]) {
                if (data[0][ix].match("\".*\""))
                    data[0][ix] = "\"s\"";
                else if (data[0][ix].match("\'.*\'"))
                    data[0][ix] = "\'s\'";
                else if (data[0][ix].match("/.*/"))
                    data[0][ix] = "/r/";
                else if (data[0][ix].match("([0-9].*|\.[0-9].*)"))
                    data[0][ix] = "0";
                data[0][ix] = data[0][ix].replace(/\\s/, "");
            }
            // Drop files with too many 'any's
            var fails = 0;
            var hits = 0;
            for (var i in data[0]) {
                if (data[0][i].match("[a-zA-Z\$\_].*") && keywords.indexOf(data[0][i]) < 0) {
                    if (data[1][i] == "$any$")
                        fails++;
                    else
                        hits++;
                }
            }
            if (fails / (hits + fails) > ANY_THRESHOLD)
                continue;
            // Produce content and double-test for inconsistencies
            var content = data[0].filter(val => val.length > 0).join(" ") + "\t" + data[1].filter(val => val.length > 0).join(" ");
            var pretend = content.split("\t");
            var left = pretend[0].split(" ");
            var right = pretend[1].split(" ");
            if (left.length != right.length)
                console.log(left.length + ", " + right.length);
            allFileTokens.push(content);
        }
        catch (e) {
            console.log(e);
            console.log("Error parsing file " + filename);
        }
    }
    return allFileTokens;
}
function extractTokens(tree, checker) {
    var texts = [[], []];
    for (var i in tree.getChildren()) {
        var ix = parseInt(i);
        var child = tree.getChildren()[ix];
        if (removableLexicalKinds.indexOf(child.kind) != -1 ||
            ts.SyntaxKind[child.kind].indexOf("JSDoc") != -1) {
            continue;
        }
        else if (templateKinds.indexOf(child.kind) != -1) {
            texts[0].push("`template`");
            texts[1].push("O");
            continue;
        }
        if (child.getChildCount() == 0) {
            var source = child.getText();
            var target = "O";
            switch (child.kind) {
                case ts.SyntaxKind.Identifier:
                    try {
                        let symbol = checker.getSymbolAtLocation(child);
                        let type = checker.typeToString(checker.getTypeOfSymbolAtLocation(symbol, child));
                        if (checker.isUnknownSymbol(symbol) || type.startsWith("typeof"))
                            target = "$any$";
                        else if (type.startsWith("\""))
                            target = "O";
                        else if (type.match("[0-9]+"))
                            target = "O";
                        else
                            target = '$' + type + '$';
                        break;
                    }
                    catch (e) { }
                    break;
                case ts.SyntaxKind.NumericLiteral:
                    target = "O";
                    break;
                case ts.SyntaxKind.StringLiteral:
                    target = "O";
                    break;
                case ts.SyntaxKind.RegularExpressionLiteral:
                    target = "O";
                    break;
            }
            target = target.trim();
            if (target.match(".+ => .+")) {
                target = "$" + target.substring(target.lastIndexOf(" => ") + 4);
            }
            if (target.match("\\s")) {
                target = "$complex$";
            }
            if (source.length == 0 || target.length == 0) {
                continue;
            }
            if (target != "O") {
                var parentKind = ts.SyntaxKind[tree.kind];
                if (parentKind.toLowerCase().indexOf("template") >= 0)
                    target = "O";
            }
            texts[0].push(source);
            texts[1].push(target);
        }
        else {
            var childTexts = extractTokens(child, checker);
            texts[0] = texts[0].concat(childTexts[0]);
            texts[1] = texts[1].concat(childTexts[1]);
        }
    }
    return texts;
}
function walkSync(dir, filelist, suffix) {
    var fs = fs || require('fs'), files = fs.readdirSync(dir);
    filelist = filelist || [];
    files.forEach(function (file) {
        let fullPath = path.join(dir, file);
        try {
            if (fs.statSync(fullPath).isDirectory()) {
                filelist = walkSync(dir + '/' + file, filelist, suffix);
            }
            else if (file.endsWith('.ts')) {
                filelist.push(fullPath);
            }
        }
        catch (e) {
            console.error("Error processing " + file);
        }
    });
    return filelist;
}
;
//# sourceMappingURL=app.js.map