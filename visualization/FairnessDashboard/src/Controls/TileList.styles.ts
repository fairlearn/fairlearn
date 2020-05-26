import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets } from "office-ui-fabric-react";

export interface ITilesListStyles {
    container: IStyle;
    itemCell: IStyle;
    iconClass: IStyle;
    title: IStyle;
    description: IStyle;
}

export const TileListStyles: () => IProcessedStyleSet<ITilesListStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<ITilesListStyles>({
        container: {
            display: "inline-flex",
            flexDirection: "row",
            flexWrap: "wrap",
            justifyContent: "space-between",
            borderBottom: "1px solid #CCCCCC",
        },
        itemCell: {
            padding: "15px",
            width: "235px",
            position: "relative",
            float: "left",
            cursor: "pointer",
            boxSizing: "border-box",
            backgroundColor: "#FFFFFF",
            marginBottom: "10px",
            marginRight: "10px",
            selectors: {
              '&:hover': { background: "lightgray" }
            }
        },
        iconClass: {
            fontSize: "20px",
            position: "absolute",
            right: "10px",
            top: "10px"
        },
        title: {
            color: "#333333",
            fontSize: "18px",
            lineHeight: "22px",
            fontWeight: "500",
            paddingRight: "16px",
            margin: 0
        },
        description: {
            paddingTop: "10px",
            color: "#666666",
            fontSize: "15px",
            lineHeight: "20px",
            fontWeight: "400"
        }
    });
};