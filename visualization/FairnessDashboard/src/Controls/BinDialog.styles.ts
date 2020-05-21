import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets } from "office-ui-fabric-react";

export interface IBinDialogStyles {
    frame: IStyle;
    header: IStyle;
    buttons: IStyle;
    saveButton: IStyle;
    cancelButton: IStyle;
    binCounter: IStyle;
    main: IStyle;
    categoryHeader: IStyle;
    checkbox: IStyle;
    controls: IStyle;
    scrollArea: IStyle;
    groupLabel: IStyle;
}

export const BinDialogStyles: () => IProcessedStyleSet<IBinDialogStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IBinDialogStyles>({
        frame: {
            height: "400px",
            width: "500px",
            display: "flex",
            flexDirection: "column",
            backgroundColor: "#F2F2F2"
        },
        header: {
            padding: "12px",
            textAlign: "center",
            backgroundColor: "#333333",
            color: "#FFFFFF",
            fontSize: "24px",
            lineHeight: "30px",
            fontWeight: "100"
        },
        buttons: {
            display: "inline-flex",
            flexDirection: "row-reverse",
            padding: "10px"
        },
        saveButton: {
            height: "44px",
            padding: "12px",
            color: "#FFFFFF",
            fontSize: "18px",
            lineHeight: "24px",
            backgroundColor: "#666666",
            fontWeight: "400",
            marginLeft: "10px"
        },
        cancelButton: {
            height: "44px",
            padding: "12px",
            color: "#333333",
            fontSize: "18px",
            lineHeight: "24px",
            backgroundColor: "#FFFFFF",
            fontWeight: "400"
        },
        binCounter: {
            selectors: {
                "& label": {
                    color: "#333333",
                    fontSize: "15px",
                    lineHeight: "20px",
                    fontWeight: "500"
                }
            }
        },
        main: {
            flexGrow: 1,
            padding: "20px 40px",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column"
        },
        categoryHeader: {

        },
        checkbox: {
            selectors: {
                "& span": {
                    color: "#333333",
                    fontSize: "15px",
                    lineHeight: "20px",
                    fontWeight: "500"
                }
            }
        },
        controls: {
            display: "inline-flex",
            width: "100%",
            justifyContent: "space-between",
            height: "30px",
            alignItems: "center",
            paddingBottom: "10px",
            borderBottom: "1px solid #CCCCCC",
            marginBottom: "10px"
        },
        scrollArea: {
            overflowY: "auto",
            overflowX: "hidden",
            flexGrow:"2"
        },
        groupLabel: {
            color: "#333333",
            fontSize: "15px",
            lineHeight: "20px",
            fontWeight: "400",
            height: "25px",
            borderBottom: "1px solid #CCCCCC",
            paddingLeft: "12px"
        }
    });
};