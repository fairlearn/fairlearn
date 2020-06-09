import { getTheme, IProcessedStyleSet, IStyle, mergeStyleSets, FontWeights, FontSizes } from "office-ui-fabric-react";

export interface IParityTabStyles {
    iconClass: IStyle;
    itemsList: IStyle;
    itemCell: IStyle;
    frame: IStyle;
    main: IStyle;
    header: IStyle;
    textBody: IStyle;
}

export const ParityTabStyles: () => IProcessedStyleSet<IParityTabStyles> = () => {
    const theme = getTheme();
    return mergeStyleSets<IParityTabStyles>({
        itemCell:
        {
            padding: "30px 36px 20px 0",
            width: "100%",
            position: "relative",
            float: "left",
            cursor: "pointer",
            boxSizing: "border-box",
            borderBottom: "1px solid #CCCCCC",
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
        itemsList: {
            overflowY: "auto"
        },
        frame: {
            width: "750px",
            height: "100%",
        },
        main: {
            height: "100%",
            minWidth: "550px",
            flex: 1
        },
        header: {
            color: "#333333",
            fontSize: "32px",
            lineHeight: "40px",
            fontWeight: "300",
            margin: "26px 0"
        },
        textBody: {
            color: "#333333",
            paddingTop: "12px",
            fontSize: "15px",
            lineHeight: "20px",
            fontWeight: "normal",
            paddingBottom: "50px"
        }
    });
};