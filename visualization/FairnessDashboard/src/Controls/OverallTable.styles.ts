import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, FontSizes } from "office-ui-fabric-react";

export interface IOverallTableStyles {
    minMaxLabel: IStyle,
    groupCol: IStyle,
    groupLabel: IStyle,
    flexCol: IStyle,
    binBox: IStyle,
    binTitle: IStyle,
    binLabel: IStyle,
    sensitiveAttributes: IStyle,
    metricCol: IStyle,
    metricLabel: IStyle,
    metricBox: IStyle,
    frame: IStyle
}

export const OverallTableStyles: () => IProcessedStyleSet<IOverallTableStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IOverallTableStyles>({
        minMaxLabel: {
            padding: "1px 9px",
            marginTop: "4px",
            color: "#FFFFFF",
            fontSize: "10px",
            lineHeight: "20px",
            fontWeight: "400",
            backgroundColor: "#999999"
        },
        groupCol: {
            display: "inline-flex",
            flexDirection:"column",
            height: "100%",
            width: "max-content"
        },
        groupLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "12px",
            fontWeight: "500",
            height: "26px"
        },
        flexCol: {
            display: "flex",
            flex: 1,
            flexDirection: "column",
            borderTop: "0.5px solid #CCCCCC",
            // borderBottom: "0.5px solid #CCCCCC"
        },
        binBox: {
            flex: 1,
            // minWidth: "100px",
            // maxWidth: "200px",
            // width: "max-content",
            width: "130px",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            borderBottom: "0.5px solid #CCCCCC"
        },
        binTitle: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "600"
        },
        binLabel: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal"
        },
        sensitiveAttributes: {
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal",
            height: "26px",
            paddingLeft: "10px"
        },
        metricCol: {
            display: "inline-flex",
            flexDirection:"column",
            height: "100%",
            width: "120px"
        },
        metricLabel: {
            color: "#333333",
            fontSize: "11px",
            lineHeight: "16px",
            fontWeight: "600",
            height: "26px",
            paddingLeft: "10px"
        },
        metricBox: {
            flex: 1,
            paddingLeft: "10px",
            color: "#333333",
            fontSize: "12px",
            lineHeight: "16px",
            fontWeight: "normal",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            // borderLeft: "0.5px dashed #CCCCCC",
            borderBottom: "0.5px solid #CCCCCC"
        },
        frame: {
            paddingBottom: "19px",
            display: "flex"
        }
    });
};